"""Counterfactual pickup decisions on audited ISS deals with a frozen card policy."""

import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from random import Random
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from iss import require_sgf_support
from iss.pipeline import connect, records, select_games, sha256
from iss.protocol import parse_record
from skat.agents.declaring import DeclarationChoice, HeuristicDeclarer
from skat.agents.neural_declaring.features import (
    FEATURE_SCHEMA,
    INPUT_SIZE,
    declaration_state,
)
from skat.agents.neural_declaring.inference import candidate_games
from skat.agents.neural_declaring.samples import FORMAT, SPLITS
from skat.agents.policy import PolicyAgent
from skat.game import Round
from skat.rules import RULES_SCHEMA
from skat.utils.reproducibility import stage_seed


class ScriptedPolicyAgent(PolicyAgent):
    """Use the frozen policy for cards and a prescribed pickup decision."""

    choice = None

    def press_skat(self):
        if self.choice is None:
            raise ValueError("missing scripted declaration")
        return self.choice.pressed

    def declare_game(self, state=None):
        if self.choice is None:
            raise ValueError("missing scripted declaration")
        return self.choice.game


_worker_agents = None


def _init_worker(policy, device, seed):
    import torch

    torch.set_num_threads(1)
    global _worker_agents
    _worker_agents = [
        ScriptedPolicyAgent(policy, device=device, seed=seed + seat)
        for seat in range(3)
    ]
    for agent in _worker_agents[1:]:
        agent.net = _worker_agents[0].net


def _simulate_record(payload, pairs_per_game, candidate_budget, seed):
    identifier, original, soloist, bid = payload
    record = SimpleNamespace(
        identifier=identifier, original=original, soloist=soloist, bid=bid
    )
    choices = sampled_choices(record, pairs_per_game, seed, candidate_budget)
    cards = set(original[soloist]) | set(original[3])
    states, labels = [], []
    for index, choice in enumerate(choices):
        kept = cards - set(choice.pressed)
        states.append(
            declaration_state(
                kept, choice.pressed, original[3], choice.game, soloist, 2
            )
        )
        for agent in _worker_agents:
            agent.choice = None
        _worker_agents[soloist].choice = choice
        round_ = Round(
            dealer=2,
            skip_bidding=True,
            solo_player_id=soloist,
            agents=_worker_agents,
            initial_cards=original,
            seed=stage_seed(seed, f"{identifier}:{index}"),
            start=False,
        )
        labels.append(float(round_.start()))
    return states, labels


def sampled_choices(record, pairs_per_game, seed, candidate_budget=None):
    cards = tuple(
        sorted(
            (*record.original[record.soloist], *record.original[3]),
            key=lambda c: c.np_index,
        )
    )
    games = candidate_games(cards, record.bid)
    pairs = list(combinations(cards, 2))
    rng = Random(stage_seed(seed, record.identifier))
    selected = {}
    for game in games:
        for pressed in rng.sample(pairs, min(pairs_per_game, len(pairs))):
            selected[
                (*[c.np_index for c in pressed], game.declaration.kind, game.suit)
            ] = DeclarationChoice(pressed, game)
    baseline_choice = None
    if games:
        baseline = HeuristicDeclarer()
        from types import SimpleNamespace

        view = SimpleNamespace(hand=cards, highest_bid=record.bid)
        choice = baseline.choose_after_pickup(view)
        if choice.game.spielwert(cards) >= record.bid:
            pressed = tuple(sorted(choice.pressed, key=lambda c: c.np_index))
            key = (
                *[c.np_index for c in pressed],
                choice.game.declaration.kind,
                choice.game.suit,
            )
            selected[key] = DeclarationChoice(pressed, choice.game)
            baseline_choice = selected[key]
    choices = list(selected.values())
    if candidate_budget is None or len(choices) <= candidate_budget:
        return choices
    if baseline_choice is None:
        return rng.sample(choices, candidate_budget)
    others = [choice for choice in choices if choice is not baseline_choice]
    return [baseline_choice, *rng.sample(others, candidate_budget - 1)]


def import_declarations(
    audit_directory,
    output,
    *,
    policy,
    max_games=2000,
    pairs_per_game=4,
    seed=7,
    device="cpu",
    progress=True,
    archive_override=None,
    candidate_budget=None,
    workers=1,
):
    """Write supervised alternatives; this function is intentionally opt-in and costly."""
    require_sgf_support()
    if pairs_per_game < 1 or pairs_per_game > 66:
        raise ValueError("pairs_per_game must be in 1..66")
    if candidate_budget is not None and candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    predicate = """ AND json_extract(metadata, '$.declaration.hand')=0
        AND json_extract(metadata, '$.declaration.ouvert')=0
        AND json_extract(metadata, '$.declaration.schneider_announced')=0
        AND json_extract(metadata, '$.declaration.schwarz_announced')=0 """
    archive, provenance = select_games(
        audit_directory,
        output,
        max_games=max_games,
        seed=seed,
        predicate=predicate,
        validation_halves=True,
        archive_override=archive_override,
    )
    output = Path(output)
    policy = Path(policy)
    states = {split: [] for split in SPLITS}
    labels = {split: [] for split in SPLITS}
    games_seen = Counter()
    status_path = output / "status.json"

    def save_status(stage):
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                dict(
                    stage=stage,
                    games=dict(games_seen),
                    rows={split: len(labels[split]) for split in SPLITS},
                ),
                indent=2,
            )
            + "\n"
        )
        temporary.replace(status_path)

    save_status("simulating")
    with (
        connect(output / "index.sqlite") as db,
        tqdm(
            total=sum(provenance["selected_games"].values()),
            desc="declaration deals",
            unit="deal",
            disable=not progress,
        ) as deal_bar,
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(policy, device, seed),
        ) as pool,
    ):
        targets = iter(db.execute("SELECT line, split FROM selection ORDER BY line"))
        target = next(targets, None)
        pending = deque()

        def consume():
            split, future = pending.popleft()
            row_states, row_labels = future.result()
            states[split].extend(row_states)
            labels[split].extend(row_labels)
            games_seen[split] += 1
            deal_bar.update()
            if sum(games_seen.values()) % 1000 == 0:
                save_status("simulating")

        with records(archive, progress=progress) as lines:
            for line_number, line in lines:
                if target is None:
                    break
                if line_number != target[0]:
                    continue
                record = parse_record(line)
                split = target[1]
                payload = (
                    record.identifier,
                    record.original,
                    record.soloist,
                    record.bid,
                )
                pending.append(
                    (
                        split,
                        pool.submit(
                            _simulate_record,
                            payload,
                            pairs_per_game,
                            candidate_budget,
                            seed,
                        ),
                    )
                )
                if len(pending) >= workers * 4:
                    consume()
                target = next(targets, None)
        if target is not None:
            raise ValueError("archive ended before selected record")
        while pending:
            consume()
    rows = {}
    checksums = {}
    for split in SPLITS:
        if not states[split]:
            raise ValueError(f"no counterfactual candidates in {split}")
        states_path = output / f"{split}-states.npy"
        labels_path = output / f"{split}-labels.npy"
        np.save(
            states_path, np.asarray(states[split], dtype=np.float32), allow_pickle=False
        )
        np.save(
            labels_path, np.asarray(labels[split], dtype=np.float32), allow_pickle=False
        )
        rows[split] = len(labels[split])
        checksums[split] = dict(states=sha256(states_path), labels=sha256(labels_path))
    manifest = dict(
        format=FORMAT,
        feature_schema=FEATURE_SCHEMA,
        input_size=INPUT_SIZE,
        rules_schema=RULES_SCHEMA,
        policy_sha256=sha256(policy),
        rows=rows,
        games=dict(games_seen),
        checksums=checksums,
        provenance=provenance,
        pairs_per_game=pairs_per_game,
        candidate_budget=candidate_budget,
        workers=workers,
        label="fixed-policy counterfactual soloist win on original ISS deal",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_status("complete")
    return manifest
