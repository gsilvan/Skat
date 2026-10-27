"""One engine writer per table; HTTP threads exchange immutable snapshots."""

import copy
import math
import threading
import time
import uuid
from dataclasses import asdict

from skat.__main__ import get_player
from skat.config import MCTSConfig
from skat.deal_code import decode_deal
from skat.game import Round
from skat.utils.reproducibility import stage_seed

from .agents import AssistedWebAgent, ObservedAgent, WebAgent
from .serialize import snapshot

KINDS = ("human", "random", "mcts", "policy")


class Conflict(ValueError):
    pass


class SessionStopped(Exception):
    pass


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: ganze Zahl zwischen {low} und {high} erforderlich")
    return value


def number(value, name, low, high):
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name}: Zahl zwischen {low} und {high} erforderlich")
    return value


def same_option(value, option):
    """JSON options require exact types, including inside declarations."""
    if type(value) is not type(option):
        return False
    if isinstance(option, dict):
        return value.keys() == option.keys() and all(
            same_option(value[key], expected) for key, expected in option.items()
        )
    return value == option


def valid_answer(kind, options, value):
    if kind == "press":
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(type(v) is int for v in value)
            and len(set(value)) == 2
            and all(v in options for v in value)
        )
    # bool is an int subclass: JSON true must never play card 1.
    return any(same_option(value, option) for option in options)


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError("Konfiguration muss ein Objekt sein")
    config = copy.deepcopy(value)
    if "dealCode" in config:
        decode_deal(config["dealCode"])
    seats = config.get("seats")
    if not isinstance(seats, list) or len(seats) != 3:
        raise ValueError("Genau drei Sitzplätze erforderlich")
    for i, seat in enumerate(seats):
        if not isinstance(seat, dict) or seat.get("kind") not in KINDS:
            raise ValueError(f"Ungültiger Spieler an Platz {i + 1}")
        seat["execution"] = seat.get(
            "execution", "manual" if seat["kind"] == "human" else "auto"
        )
        if seat["execution"] not in ("auto", "manual") or (
            seat["kind"] == "human" and seat["execution"] != "manual"
        ):
            raise ValueError(f"Ungültige Ausführung an Platz {i + 1}")
        label = seat.get("label") or f"Spieler {i + 1}"
        if not isinstance(label, str) or len(label) > 80:
            raise ValueError("Spielername darf höchstens 80 Zeichen enthalten")
        seat["label"] = label
        for key, default, choices in [
            ("bidding", "conservative", ("conservative", "neural")),
            ("declaration", "heuristic", ("heuristic", "neural")),
            ("pickup", "always", ("always", "neural")),
        ]:
            seat[key] = seat.get(key, default)
            if seat[key] not in choices:
                raise ValueError(f"Ungültige Strategie: {key}")
        for key in (
            "policy",
            "biddingModel",
            "declarationModel",
            "pickupModel",
            "beliefModel",
        ):
            path = seat.get(key)
            if path is not None and (not isinstance(path, str) or len(path) > 4096):
                raise ValueError(f"Ungültiger Modellpfad: {key}")
        if seat["kind"] != "human":
            required = ["policy"] if seat["kind"] == "policy" else []
            required += [
                key + "Model"
                for key in ("bidding", "declaration", "pickup")
                if seat[key] == "neural"
            ]
            for key in required:
                if not seat.get(key):
                    raise ValueError(f"Platz {i + 1}: Modell für {key} fehlt")
        seat["biddingThreshold"] = number(
            seat.get("biddingThreshold", 2 / 3), "Reizschwelle", 0, 1
        )
        seat["pickupMargin"] = number(seat.get("pickupMargin", 0), "Hand-Abstand", 0, 1)
    config["seed"] = integer(config.get("seed", 42), "Seed", 0, 2**32 - 1)
    config["dealer"] = integer(config.get("dealer", 0), "Geber", 0, 2)
    config["speedMs"] = integer(config.get("speedMs", 450), "Tempo", 0, 5000)
    workers = config.get("workers")
    config["workers"] = None if workers is None else integer(workers, "Worker", 1, 1024)
    config["simulations"] = integer(
        config.get("simulations", 600), "Simulationen", 1, 1000000
    )
    config["determinizations"] = integer(
        config.get("determinizations", 20), "Welten", 1, config["simulations"]
    )
    config["device"] = config.get("device", "cpu")
    if config["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("Gerät muss cpu, cuda oder auto sein")
    for key in ("reveal", "paused"):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f"{key} muss ein Wahrheitswert sein")
    return config


class GameSession:
    def __init__(self, config):
        self.config = validate_config(config)
        self.imported_deal = (
            decode_deal(self.config.pop("dealCode"))
            if "dealCode" in self.config
            else None
        )
        if self.imported_deal is not None:
            self.config["dealer"] = self.imported_deal.dealer
        self.id = uuid.uuid4().hex
        self.changed = threading.Condition(threading.RLock())
        self.status = "created"
        self.revision = 0
        self.round = None
        self.thread = None
        self.pending = None
        self.answer = None
        self.active = None
        self.last_action = None
        self.error = None
        self.paused = self.config.get("paused", False)
        self.reveal = self.config.get("reveal", False)
        self.speed_ms = self.config["speedMs"]
        self.steps = 0
        self.stopped = False
        self.round_index = 0
        self.records = []
        self.comparisons = []
        self.comparison_stats = [
            {"seat": seat, "matches": 0, "total": 0, "byKind": {}}
            for seat, config in enumerate(self.config["seats"])
            if config["kind"] != "human" and config["execution"] == "manual"
        ]
        self.scores = [0, 0, 0]
        self.cached = ({}, {})
        self.agents = None
        self.pool = None
        self.last_access = time.monotonic()

    def touch(self):
        """Called under the condition; never inspects engine state."""
        self.revision += 1
        self.changed.notify_all()

    def publish(self):
        """Only the engine thread may serialize the live engine."""
        with self.changed:
            self.cached = (snapshot(self), snapshot(self, reveal=True))
            self.touch()

    def get_snapshot(self, after=None, timeout=20):
        with self.changed:
            self.last_access = time.monotonic()
            if after is not None:
                self.changed.wait_for(
                    lambda: (
                        self.revision != after or self.status in {"stopped", "error"}
                    ),
                    timeout,
                )
            state = copy.deepcopy(self.cached[int(self.reveal)])
            state.update(
                {
                    "id": self.id,
                    "revision": self.revision,
                    "status": self.status,
                    "config": self.config,
                    "pending": self.pending,
                    "active": self.active,
                    "lastAction": self.last_action,
                    "error": self.error,
                    "paused": self.paused,
                    "reveal": self.reveal,
                    "speedMs": self.speed_ms,
                    "roundIndex": self.round_index,
                    "records": self.records,
                    "comparisons": self.comparisons,
                    "comparisonStats": self.comparison_stats,
                    "scores": self.scores,
                }
            )
            return copy.deepcopy(state)

    def start(self):
        with self.changed:
            if self.thread is not None:
                raise Conflict("Partie wurde bereits gestartet")
            self.status = "loading"
            self.touch()
            self.thread = threading.Thread(
                target=self.run, daemon=True, name=f"skat-web-{self.id[:8]}"
            )
            self.thread.start()

    def make_agents(self):
        from skat.agents.search.parallel import SearchPool

        self.pool = SearchPool(
            self.config["workers"], capacity=self.config["determinizations"]
        )
        agents = []
        for seat, config in enumerate(self.config["seats"]):
            seed = stage_seed(self.config["seed"], f"web-agent-{seat}")
            if config["kind"] == "human":
                agents.append(WebAgent(self, seat, seed))
                continue
            device = self.config["device"]
            bidder = declarer = pickup = None
            if config["bidding"] == "neural":
                from skat.agents.neural_bidding.inference import NeuralBidder

                bidder = NeuralBidder(
                    config["biddingModel"],
                    device=device,
                    threshold=config["biddingThreshold"],
                )
            if config["declaration"] == "neural":
                from skat.agents.neural_declaring.inference import DeclarationScorer

                declarer = DeclarationScorer(config["declarationModel"], device=device)
            if config["pickup"] == "neural":
                from skat.agents.neural_pickup.inference import PickupScorer

                pickup = PickupScorer(config["pickupModel"], device=device)
            agent = get_player(
                config["kind"],
                seed=seed,
                policy=config.get("policy"),
                device=device,
                mcts=MCTSConfig(
                    self.config["simulations"], self.config["determinizations"]
                ),
                search_pool=self.pool,
                bidding=config["bidding"],
                bidder=bidder,
                belief_model=config.get("beliefModel")
                if config["kind"] == "mcts"
                else None,
                declaration=config["declaration"],
                declaration_model=declarer,
                pickup=config["pickup"],
                pickup_model=pickup,
                pickup_margin=config["pickupMargin"],
            )
            agents.append(
                AssistedWebAgent(agent, self, seat, seed)
                if config["execution"] == "manual"
                else ObservedAgent(agent, self, seat)
            )
        return agents

    def run(self):
        try:
            self.agents = self.make_agents()
            while True:
                with self.changed:
                    self.check_stopped()
                    self.round = Round(
                        dealer=(self.config["dealer"] + self.round_index) % 3,
                        agents=self.agents,
                        initial_cards=self.imported_deal.initial_cards
                        if self.imported_deal is not None and self.round_index == 0
                        else None,
                        seed=stage_seed(
                            self.config["seed"], f"web-deal-{self.round_index}"
                        ),
                        start=False,
                    )
                    self.status = "running"
                self.publish()
                self.round.start()
                with self.changed:
                    self.check_stopped()
                    result = asdict(self.round.result)
                    self.records.append(
                        {
                            "round": self.round_index + 1,
                            "dealer": self.round.dealer,
                            "dealCode": self.round.deal_code,
                            "result": result,
                        }
                    )
                    if result["soloist"] is not None:
                        self.scores[result["soloist"]] += result["score"]
                    self.status = (
                        "passed" if result["soloist_won"] is None else "finished"
                    )
                    self.active = None
                    self.publish()
                with self.changed:
                    self.changed.wait_for(
                        lambda: self.status == "loading" or self.stopped
                    )
                    self.check_stopped()
                    self.round_index += 1
                    self.round = None
                    self.cached = ({}, {})
                    self.last_action = None
                    self.touch()
        except SessionStopped:
            with self.changed:
                self.status = "stopped"
        except Exception as error:
            with self.changed:
                self.status = "error"
                self.error = f"{type(error).__name__}: {error}"
        finally:
            if self.pool is not None:
                self.pool.close()
            with self.changed:
                self.pending = self.active = None
                self.touch()

    def check_stopped(self):
        if self.stopped:
            raise SessionStopped()

    def ask(self, seat, kind, options, **details):
        with self.changed:
            self.check_stopped()
            if "suggestion" in details and not valid_answer(
                kind, options, details["suggestion"]
            ):
                raise ValueError("Agentenvorschlag ist für diese Entscheidung ungültig")
            prompt = {
                "id": uuid.uuid4().hex,
                "seat": seat,
                "kind": kind,
                "options": options,
                **details,
            }
            self.pending, self.answer, self.active = prompt, None, None
            self.publish()
            self.changed.wait_for(lambda: self.answer is not None or self.stopped)
            self.check_stopped()
            answer = self.answer[0]
            if "suggestion" in prompt:
                suggestion = prompt["suggestion"]
                matches = (
                    set(answer) == set(suggestion)
                    if kind == "press"
                    else same_option(answer, suggestion)
                )
                self.comparisons.append(
                    {
                        "decisionId": prompt["id"],
                        "roundIndex": self.round_index,
                        "seat": seat,
                        "kind": kind,
                        "suggestion": copy.deepcopy(suggestion),
                        "actual": copy.deepcopy(answer),
                        "matches": matches,
                    }
                )
                stats = next(s for s in self.comparison_stats if s["seat"] == seat)
                by_kind = stats["byKind"].setdefault(kind, {"matches": 0, "total": 0})
                for counter in (stats, by_kind):
                    counter["total"] += 1
                    counter["matches"] += int(matches)
            self.pending = self.answer = None
            self.last_action = {
                "seat": seat,
                "kind": kind,
                "value": None if kind == "press" else answer,
            }
            self.touch()
            return answer

    def submit(self, decision_id, value):
        with self.changed:
            if (
                self.stopped
                or self.pending is None
                or self.pending["id"] != decision_id
                or self.answer is not None
            ):
                raise Conflict("Diese Entscheidung ist nicht mehr offen")
            options = self.pending["options"]
            if not valid_answer(self.pending["kind"], options, value):
                raise ValueError("Ungültige Aktion für diese Entscheidung")
            self.answer = (copy.deepcopy(value),)
            self.touch()

    def before_decision(self, seat, kind, *, manual=False):
        with self.changed:
            self.check_stopped()
            self.active = {"seat": seat, "kind": kind}
            self.publish()
            if manual:
                return
            self.changed.wait_for(
                lambda: not self.paused or self.steps > 0 or self.stopped
            )
            self.check_stopped()
            if self.paused:
                self.steps -= 1
            deadline = time.monotonic() + self.speed_ms / 1000
            self.changed.wait_for(
                lambda: self.stopped, max(0, deadline - time.monotonic())
            )
            self.check_stopped()

    def after_decision(self, seat, kind, value):
        with self.changed:
            self.check_stopped()
            self.last_action = {
                "seat": seat,
                "kind": kind,
                "value": None if kind == "press" else value,
            }
            self.active = None
            self.touch()

    def control(self, command, value=None, round_index=None):
        if not isinstance(command, str):
            raise ValueError("Steuerbefehl muss eine Zeichenkette sein")
        with self.changed:
            if self.stopped:
                raise Conflict("Partie wurde beendet")
            if command == "next":
                if (
                    self.status not in {"finished", "passed"}
                    or type(round_index) is not int
                    or round_index != self.round_index
                ):
                    raise Conflict("Nächste Runde ist noch nicht verfügbar")
                self.status = "loading"
            elif command in {"pause", "resume", "step"}:
                self.paused = command != "resume"
                self.steps = self.steps + 1 if command == "step" else 0
            elif command == "speed":
                self.speed_ms = integer(value, "Tempo", 0, 5000)
            elif command == "reveal":
                if type(value) is not bool:
                    raise ValueError("Sichtbarkeit muss ein Wahrheitswert sein")
                self.reveal = value
            else:
                raise ValueError("Unbekannter Steuerbefehl")
            self.touch()

    def stop(self):
        with self.changed:
            self.stopped = True
            self.status = "stopped"
            self.pending = None
            self.touch()


class SessionRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}

    def create(self, config):
        session = GameSession(config)
        with self.lock:
            for key, old in list(self.sessions.items()):
                if time.monotonic() - old.last_access > 3600:
                    old.stop()
                    del self.sessions[key]
            if len(self.sessions) >= 16:
                raise Conflict("Zu viele offene Partien; alte Partien zuerst beenden")
            self.sessions[session.id] = session
        session.start()
        return session

    def get(self, key):
        with self.lock:
            return self.sessions[key]

    def remove(self, key):
        with self.lock:
            session = self.sessions.pop(key)
        session.stop()

    def close(self):
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.stop()
        for session in sessions:
            if session.thread is not None:
                session.thread.join(timeout=5)


def discover_checkpoints(root):
    """List local candidates; existing strict loaders check their actual schemas."""
    paths = set()
    for directory in (root / "trained_models", root / "benchmark_runs"):
        paths.update(str(path.relative_to(root)) for path in directory.rglob("*.pt"))
    return sorted(paths)
