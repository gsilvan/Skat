"""Paired pickup comparison with reference RNGs reset for every deal.

This is separate from the fixed README rating protocol. It prevents an early
Hand game from shifting Random opponents' card draws in later rounds.
"""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

import torch

from skat.__main__ import get_player
from skat.agents.neural_declaring.inference import DeclarationScorer
from skat.agents.neural_pickup.inference import PickupScorer
from skat.game import Round
from skat.utils.reproducibility import stage_seed


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(policy, pickup_model, declaration_model, *, rounds=2000, seed=43):
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be positive")
    torch.set_num_threads(1)
    pickup_scorer = PickupScorer(pickup_model)
    declaration_scorer = DeclarationScorer(declaration_model)
    data = dict(
        protocol="paired-seat0-random-reseeded-v1",
        rounds=rounds,
        seed=seed,
        policy_sha256=digest(policy),
        pickup_model_sha256=digest(pickup_model),
        declaration_model_sha256=digest(declaration_model),
        comparisons={},
    )
    for declaration in ("heuristic", "neural"):
        candidates = [
            get_player(
                "policy",
                policy=policy,
                declaration=declaration,
                declaration_model=declaration_scorer,
                pickup=pickup,
                pickup_model=pickup_scorer,
            )
            for pickup in ("always", "neural")
        ]
        rows = []
        for index in range(rounds):
            deal_seed = stage_seed(seed, f"pickup-deal-{index}")
            outcomes = []
            for candidate in candidates:
                agents = [
                    candidate,
                    get_player("random", seed=0),
                    get_player("random", seed=0),
                ]
                for seat, agent in enumerate(agents):
                    agent.rng = Random(stage_seed(deal_seed, f"seat-{seat}"))
                round_ = Round(
                    dealer=index % 3, agents=agents, seed=deal_seed, start=False
                )
                soloist_won = round_.start()
                candidate_won = (
                    None
                    if soloist_won is None
                    else bool(soloist_won)
                    if round_.solo_player_id == 0
                    else not soloist_won
                )
                outcomes.append(
                    dict(
                        candidate_won=candidate_won,
                        soloist=round_.solo_player_id
                        if soloist_won is not None
                        else None,
                        soloist_won=soloist_won,
                        hand=bool(round_.game.declaration.hand)
                        if soloist_won is not None
                        else None,
                        candidate_points=round_.result.score
                        if round_.solo_player_id == 0 and soloist_won is not None
                        else 0,
                    )
                )
            rows.append(
                dict(deal=index, seed=deal_seed, always=outcomes[0], neural=outcomes[1])
            )
            if (index + 1) % 100 == 0:
                print(f"{declaration}: {index + 1}/{rounds}", flush=True)
        data["comparisons"][declaration] = dict(
            rows=rows,
            summary={
                style: dict(
                    party_wins=sum(r[style]["candidate_won"] is True for r in rows),
                    solo_wins=sum(
                        r[style]["soloist"] == 0 and r[style]["soloist_won"]
                        for r in rows
                    ),
                    hand_games=sum(
                        r[style]["soloist"] == 0 and r[style]["hand"] for r in rows
                    ),
                    points=sum(r[style]["candidate_points"] for r in rows),
                )
                for style in ("always", "neural")
            },
        )
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--pickup-model", required=True)
    parser.add_argument("--declaration-model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("use a new results path")
    result = compare(
        args.policy,
        args.pickup_model,
        args.declaration_model,
        rounds=args.rounds,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(
        json.dumps(
            {k: v["summary"] for k, v in result["comparisons"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
