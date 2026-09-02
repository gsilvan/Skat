from __future__ import annotations

import copy
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from skat.__main__ import get_player
from skat.agents import Agent
from skat.agents.bidding import BiddingStyle
from skat.agents.neural_bidding import NeuralBidder
from skat.game import Round

from .agents import ObservedAgent, WebAgent
from .serialize import snapshot

AGENT_KINDS = ("human", "random", "mcts", "belief-mcts", "policy", "dqn")


class SessionStopped(RuntimeError):
    pass


class GameSession:
    """One local round, shared by Flask request and engine worker threads."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex
        self.config = self._validate_config(config)
        self.public_config = copy.deepcopy(self.config)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.round: Round | None = None
        self.thread: threading.Thread | None = None
        self.status = "created"
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.pending: dict[str, Any] | None = None
        self.pending_answer: Any = None
        self.active_action: dict[str, Any] | None = None
        self.last_action: dict[str, Any] | None = None
        self.revision = 0
        self.paused = bool(self.config.get("paused", False))
        self.step_budget = 0
        self.speed_ms = int(self.config.get("speedMs", 450))
        self.reveal = bool(self.config.get("reveal", False))
        self.stopped = False
        self.cached_snapshot: dict[str, Any] = {}
        self.seat_kinds = [seat["kind"] for seat in self.config["seats"]]
        self.seat_labels = [
            seat.get("label") or seat["kind"] for seat in self.config["seats"]
        ]
        with self.lock:
            self._publish_locked()

    @staticmethod
    def _validate_config(value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("session configuration must be an object")
        seats = value.get("seats")
        if not isinstance(seats, list) or len(seats) != 3:
            raise ValueError("exactly three seats are required")
        normalized = copy.deepcopy(value)
        for index, seat in enumerate(seats):
            if not isinstance(seat, dict) or seat.get("kind") not in AGENT_KINDS:
                raise ValueError(f"seat {index} has an unknown agent kind")
        normalized["seed"] = int(value.get("seed", 42))
        normalized["dealer"] = int(value.get("dealer", 0)) % 3
        normalized["device"] = str(value.get("device", "cpu"))
        normalized["simulations"] = max(1, int(value.get("simulations", 600)))
        normalized["determinizations"] = max(1, int(value.get("determinizations", 20)))
        normalized["speedMs"] = max(0, min(5000, int(value.get("speedMs", 450))))
        return normalized

    def start(self) -> None:
        with self.lock:
            if self.thread is not None:
                raise RuntimeError("session already started")
            self.status = "loading"
            self._publish_locked()
            self.thread = threading.Thread(
                target=self._run,
                name=f"skat-web-{self.id[:8]}",
                daemon=True,
            )
            self.thread.start()

    def _make_agent(self, seat: int, config: dict[str, Any]) -> Agent:
        kind = config["kind"]
        seed = self.config["seed"] + seat
        if kind == "human":
            return WebAgent(self, seat, seed=seed)

        try:
            bidding = BiddingStyle(config.get("bidding", "conservative"))
        except ValueError as error:
            raise ValueError(f"seat {seat} has invalid bidding style") from error
        neural_bidder = None
        if bidding is BiddingStyle.NEURAL:
            model = config.get("biddingModel") or self.config.get("biddingModel")
            if not model:
                raise ValueError(f"seat {seat} uses neural bidding without a model")
            neural_bidder = NeuralBidder(
                str(model),
                device=self.config["device"],
                threshold=float(config.get("biddingThreshold", 0.5)),
            )

        path = config.get("checkpoint")
        agent = get_player(
            kind,
            training=False,
            path=path,
            simulations=self.config["simulations"],
            determinizations=self.config["determinizations"],
            seed=seed,
            device=self.config["device"],
            dqn_arch=config.get("architecture"),
            bidding=bidding,
            policy_path=path or "./trained_models/policy.pt",
            policy_temperature=float(config.get("temperature", 0.0)),
            belief_path=path or "./trained_models/belief.pt",
            neural_bidder=neural_bidder,
        )
        if agent is None:
            raise ValueError(f"could not construct agent {kind!r}")
        return ObservedAgent(agent, self, seat)

    def _run(self) -> None:
        agents: list[Agent] = []
        try:
            agents = [
                self._make_agent(index, seat)
                for index, seat in enumerate(self.config["seats"])
            ]
            round_ = Round(
                dealer=self.config["dealer"],
                agents=agents,
                seed=self.config["seed"],
                start=False,
            )
            with self.lock:
                self.round = round_
                self.status = "running"
                self._publish_locked()
            won = round_.start()
            with self.lock:
                if self.stopped:
                    self.status = "stopped"
                elif won is None:
                    self.status = "passed"
                    self.result = {"soloistWon": None, "message": "All players passed."}
                else:
                    self.status = "finished"
                    self.result = {
                        "soloistWon": won,
                        "soloist": round_.solo_player_id,
                        "message": "Soloist won." if won else "Defenders won.",
                    }
                self.pending = None
                self.active_action = None
                self._publish_locked()
        except SessionStopped:
            with self.lock:
                self.status = "stopped"
                self.pending = None
                self.active_action = None
                self._publish_locked()
        except Exception as error:  # noqa: BLE001 - surfaced to the local UI
            with self.lock:
                self.status = "error"
                self.error = f"{type(error).__name__}: {error}"
                self.pending = None
                self.active_action = None
                self._publish_locked()
        finally:
            for agent in agents:
                if hasattr(agent, "close"):
                    agent.close()  # type: ignore[attr-defined]

    def _publish_locked(self) -> None:
        self.revision += 1
        self.cached_snapshot = snapshot(self)
        self.changed.notify_all()

    def publish(self) -> None:
        with self.lock:
            self._publish_locked()

    def get_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.cached_snapshot)

    def wait_for_update(self, revision: int, timeout: float = 25.0) -> dict[str, Any]:
        with self.changed:
            self.changed.wait_for(
                lambda: self.revision != revision or self.stopped,
                timeout=timeout,
            )
            return copy.deepcopy(self.cached_snapshot)

    def ask_human(self, seat: int, prompt: dict[str, Any]) -> Any:
        with self.changed:
            if self.stopped:
                raise SessionStopped()
            self.pending_answer = None
            self.pending = {"seat": seat, **prompt}
            self.active_action = None
            self._publish_locked()
            self.changed.wait_for(
                lambda: self.pending_answer is not None or self.stopped
            )
            if self.stopped:
                raise SessionStopped()
            answer = self.pending_answer
            self.pending_answer = None
            self.pending = None
            self.last_action = {"seat": seat, "kind": prompt["kind"], "value": answer}
            self._publish_locked()
            return answer

    def submit_action(self, value: Any) -> None:
        with self.changed:
            if self.pending is None:
                raise ValueError("the game is not waiting for a human action")
            options = self.pending.get("options", [])
            if self.pending["kind"] == "press":
                if (
                    not isinstance(value, list)
                    or len(value) != self.pending.get("count", 2)
                    or len(set(value)) != len(value)
                    or any(card not in options for card in value)
                ):
                    raise ValueError(
                        "select exactly two different cards from your hand"
                    )
            elif value not in options:
                raise ValueError(f"action {value!r} is not currently legal")
            self.pending_answer = value
            self.changed.notify_all()

    def before_agent_action(
        self, seat: int, kind: str, details: dict[str, Any]
    ) -> None:
        with self.changed:
            if self.stopped:
                raise SessionStopped()
            self.active_action = {"seat": seat, "kind": kind, **details}
            self._publish_locked()
            self.changed.wait_for(
                lambda: not self.paused or self.step_budget > 0 or self.stopped
            )
            if self.stopped:
                raise SessionStopped()
            if self.paused and self.step_budget > 0:
                self.step_budget -= 1
            delay = self.speed_ms / 1000.0
        if delay:
            time.sleep(delay)

    def after_agent_action(self, seat: int, kind: str, value: Any) -> None:
        with self.lock:
            self.last_action = {"seat": seat, "kind": kind, "value": value}
            self.active_action = None
            self._publish_locked()

    def control(self, command: str, value: Any = None) -> None:
        with self.changed:
            if command == "pause":
                self.paused = True
            elif command == "resume":
                self.paused = False
                self.step_budget = 0
            elif command == "step":
                self.paused = True
                self.step_budget += 1
            elif command == "speed":
                self.speed_ms = max(0, min(5000, int(value)))
            elif command == "reveal":
                self.reveal = bool(value)
            else:
                raise ValueError(f"unknown control command {command!r}")
            self._publish_locked()
            self.changed.notify_all()

    def stop(self) -> None:
        with self.changed:
            self.stopped = True
            self.changed.notify_all()
            self._publish_locked()


class SessionRegistry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions: dict[str, GameSession] = {}

    def create(self, config: dict[str, Any]) -> GameSession:
        session = GameSession(config)
        with self.lock:
            self.sessions[session.id] = session
        session.start()
        return session

    def get(self, session_id: str) -> GameSession:
        with self.lock:
            try:
                return self.sessions[session_id]
            except KeyError as error:
                raise KeyError(f"unknown session {session_id}") from error


def discover_checkpoints(root: Path) -> dict[str, list[str]]:
    result = {"policy": [], "belief": [], "bidding": [], "dqn": []}
    patterns = {
        "policy": "policy-*.pt",
        "belief": "belief.pt",
        "bidding": "bidding.pt",
        "dqn": "dqn-*.pt",
    }
    directories = sorted((root / "benchmark_runs").glob("*/checkpoints"), reverse=True)
    directories.append(root / "trained_models")
    for kind, pattern in patterns.items():
        seen = set()
        for directory in directories:
            for path in sorted(directory.glob(pattern)):
                relative = str(path.relative_to(root))
                if relative not in seen:
                    result[kind].append(relative)
                    seen.add(relative)
    return result
