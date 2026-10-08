"""Evaluate exact bonus tie-breaking against a saved reward comparison.

Run this on the same device/software stack and with the same opponent policy
as the baseline comparison. Baselines are reused rather than recomputed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from random import Random

import torch

from skat.agents.mcts import MCTSAgent
from skat.agents.policy import PolicyAgent
from skat.agents.random import RandomAgent
from skat.benchmark.compare import _outcomes
from skat.benchmark.evaluation import evaluate_candidate
from skat.rewards import MCTS_TIEBREAK_OBJECTIVE
from skat.utils.reproducibility import seed_everything


def paired_comparison(old: dict, new: dict) -> dict:
    a, b = _outcomes(old), _outcomes(new)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError("No common deal/seat/panel outcomes")
    groups = {}
    for key in keys:
        groups.setdefault(key[2], []).append(int(b[key]) - int(a[key]))
    values = [(sum(group), len(group)) for group in groups.values()]
    rng = Random(42)
    bootstrap = []
    for _ in range(10_000):
        sample = rng.choices(values, k=len(values))
        bootstrap.append(sum(total for total, _ in sample) / sum(n for _, n in sample))
    bootstrap.sort()
    return {
        "old": old["name"],
        "new": new["name"],
        "paired": len(keys),
        "delta": sum(int(b[key]) - int(a[key]) for key in keys) / len(keys),
        "improved": sum(not a[key] and b[key] for key in keys),
        "regressed": sum(a[key] and not b[key] for key in keys),
        "cluster_bootstrap_95": [bootstrap[250], bootstrap[9749]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--opponent-policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    log_path = args.out.with_suffix(".log")
    if args.out.exists() or log_path.exists():
        parser.error("Output or log already exists; use a fresh --out path")
    baseline = json.loads(args.baseline.read_text())
    first = baseline["results"][0]
    blocks = first["by_seat"]
    rounds = len(blocks[0]["outcomes"])
    if len(blocks) != 6 or {b["opponent"] for b in blocks} != {
        "random",
        "policy-transformer",
    }:
        parser.error("Expected the two-panel, three-seat reward comparison")
    seed = baseline["seed"]
    torch.set_num_threads(1)
    seed_everything(seed)
    panels = {
        "random": lambda seed, seat: RandomAgent(seed=seed),
        "policy-transformer": lambda seed, seat: PolicyAgent(
            path=str(args.opponent_policy), device=args.device, seed=seed
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Evaluating exact tie-breaking: {rounds} deals × 3 seats × 2 panels",
        flush=True,
    )
    with (
        log_path.open("x") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        result = evaluate_candidate(
            "win-first-tiebreak",
            lambda seed, seat: MCTSAgent(
                seed=seed, simulations=600, determinizations=20, bonus_tiebreak=True
            ),
            rounds=rounds,
            seed=seed,
            opponents=panels,
        )
    comparisons = [paired_comparison(old, result) for old in baseline["results"]]
    payload = {
        "seed": seed,
        "rounds_per_seat_panel": rounds,
        "simulations": 600,
        "determinizations": 20,
        "device": args.device,
        "torch": torch.__version__,
        "reward_objective": MCTS_TIEBREAK_OBJECTIVE,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "opponent_policy_sha256": hashlib.sha256(
            args.opponent_policy.read_bytes()
        ).hexdigest(),
        "results": [*baseline["results"], result],
        "comparisons": comparisons,
    }
    with args.out.open("x") as out:
        json.dump(payload, out, indent=2)
    for row in payload["results"]:
        print(
            f"{row['name']}: {row['win_rate']:.2%}, "
            f"Schneider {row['schneider_wins']}, points {row['card_points_per_round']:.2f}"
        )
    for comparison in comparisons:
        print(json.dumps(comparison))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
