"""Paired Hand/pickup outcomes on audited ISS deals under one frozen policy."""

import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from iss import require_sgf_support
from iss.declaration import ScriptedPolicyAgent
from iss.pipeline import connect, records, select_games, sha256
from iss.protocol import parse_record
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.neural_pickup.features import FEATURE_SCHEMA, INPUT_SIZE, pickup_state
from skat.agents.neural_pickup.samples import FORMAT, SPLITS
from skat.game import Round
from skat.rules import RULES_SCHEMA
from skat.utils.reproducibility import stage_seed

_agents = None
_declarer = None


def _init_worker(policy, declaration_model, device, seed):
    import torch

    torch.set_num_threads(1)
    global _agents, _declarer
    _agents = [
        ScriptedPolicyAgent(policy, device=device, seed=seed + seat)
        for seat in range(3)
    ]
    for agent in _agents[1:]:
        agent.net = _agents[0].net
    if declaration_model:
        from skat.agents.neural_declaring.inference import DeclarationScorer

        _declarer = DeclarationScorer(declaration_model, device=device)
    else:
        _declarer = None


def _simulate(payload, seed):
    identifier, original, soloist, bid, historical_hand, historical_won = payload
    hand = set(original[soloist])
    feature = pickup_state(hand, bid, soloist, 2)
    view = SimpleNamespace(
        hand=tuple(hand),
        skat=original[3],
        highest_bid=bid,
        seat_id=soloist,
        dealer=2,
    )
    declarer = HeuristicDeclarer()
    if _declarer is not None:
        from skat.agents.neural_declaring.inference import NeuralDeclarer

        declarer = NeuralDeclarer(_declarer)
    pickup_view = SimpleNamespace(
        **{**vars(view), "hand": tuple((*hand, *original[3]))}
    )
    pickup_choice = declarer.choose_after_pickup(pickup_view)
    hand_game = HeuristicDeclarer().choose_hand(view)
    labels = []
    for hand_mode, choice in ((False, pickup_choice), (True, None)):
        for agent in _agents:
            agent.choice = None
        _agents[soloist].choice = (
            SimpleNamespace(game=hand_game) if hand_mode else choice
        )
        round_ = Round(
            dealer=2,
            skip_bidding=True,
            solo_player_id=soloist,
            hand_game=hand_mode,
            agents=_agents,
            initial_cards=original,
            seed=stage_seed(seed, identifier),
            start=False,
        )
        # Skipped bidding otherwise defaults to zero, masking overbids.
        round_.highest_bid = bid
        labels.append(float(round_.start()))
    return feature, labels, historical_hand, historical_won


def import_pickup(
    audit_directory,
    output,
    *,
    policy,
    declaration_model=None,
    max_games=2000,
    seed=7,
    device="cpu",
    workers=1,
    progress=True,
    archive_override=None,
):
    """Generate paired labels; history selects deals but never serves as truth."""
    require_sgf_support()
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    predicate = """ AND json_extract(metadata, '$.declaration.ouvert')=0
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
    states = {split: [] for split in SPLITS}
    labels = {split: [] for split in SPLITS}
    history = Counter()
    seen = Counter()
    status = output / "status.json"

    def save_status(stage):
        temporary = status.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                dict(stage=stage, games=dict(seen), historical=dict(history)),
                indent=2,
            )
            + "\n"
        )
        temporary.replace(status)

    save_status("simulating")
    with (
        connect(output / "index.sqlite") as db,
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(policy, declaration_model, device, seed),
        ) as pool,
        tqdm(
            total=sum(provenance["selected_games"].values()),
            desc="pickup deals",
            unit="deal",
            disable=not progress,
        ) as bar,
    ):
        targets = iter(db.execute("SELECT line, split FROM selection ORDER BY line"))
        target = next(targets, None)
        pending = deque()

        def consume():
            split, future = pending.popleft()
            state, wins, historical_hand, historical_won = future.result()
            states[split].append(state)
            labels[split].append(wins)
            history[f"{split}_hand_{int(historical_hand)}"] += 1
            history[f"{split}_won_{int(historical_won)}"] += 1
            seen[split] += 1
            bar.update()
            if sum(seen.values()) % 1000 == 0:
                save_status("simulating")

        with records(archive, progress=progress) as lines:
            for line_number, line in lines:
                if target is None:
                    break
                if line_number != target[0]:
                    continue
                record = parse_record(line)
                payload = (
                    record.identifier,
                    record.original,
                    record.soloist,
                    record.bid,
                    record.contract.hand,
                    record.won,
                )
                pending.append((target[1], pool.submit(_simulate, payload, seed)))
                if len(pending) >= workers * 4:
                    consume()
                target = next(targets, None)
        if target is not None:
            raise ValueError("archive ended before selected record")
        while pending:
            consume()
    checksums = {}
    for split in SPLITS:
        if not states[split]:
            raise ValueError(f"no pickup examples in {split}")
        state_path = output / f"{split}-states.npy"
        label_path = output / f"{split}-labels.npy"
        np.save(state_path, np.asarray(states[split], dtype=np.float32))
        np.save(label_path, np.asarray(labels[split], dtype=np.float32))
        checksums[split] = dict(states=sha256(state_path), labels=sha256(label_path))
    manifest = dict(
        format=FORMAT,
        feature_schema=FEATURE_SCHEMA,
        input_size=INPUT_SIZE,
        rules_schema=RULES_SCHEMA,
        policy_sha256=sha256(policy),
        declaration_model_sha256=sha256(declaration_model)
        if declaration_model
        else None,
        provenance=provenance,
        rows=dict(seen),
        historical=dict(history),
        checksums=checksums,
        label="paired fixed-policy wins, pickup and heuristic Hand",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_status("complete")
    return manifest
