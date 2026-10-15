"""Paired comparison of uniform and belief MCTS on identical evaluation deals."""

import argparse
import json
from pathlib import Path

from skat.evaluation import paired_comparison, paired_points_comparison, summarize


def compare(uniform_path, belief_path):
    uniform, belief = (
        json.loads(Path(path).read_text()) for path in (uniform_path, belief_path)
    )
    required = (
        "rules_schema",
        "observation_schema",
        "source_sha256",
        "seed",
        "deals",
        "deal_offset",
        "opponent_policy_sha256",
        "mcts",
        "mcts_rng_scheme",
    )
    for key in required:
        if uniform["manifest"].get(key) != belief["manifest"].get(key):
            raise ValueError(f"evaluations differ in {key}")
    if uniform["manifest"].get("belief_model_sha256") is not None:
        raise ValueError("baseline must use the uniform sampler")
    if belief["manifest"].get("belief_model_sha256") is None:
        raise ValueError("candidate must use a belief checkpoint")
    report = {}
    for panel in ("random", "policy"):
        old = [
            row
            for row in uniform["outcomes"]
            if row["agent"] == "mcts" and row["panel"] == panel
        ]
        new = [
            row
            for row in belief["outcomes"]
            if row["agent"] == "mcts" and row["panel"] == panel
        ]
        expected = 3 * uniform["manifest"]["deals"]
        if len(old) != expected or len(new) != expected:
            raise ValueError("incomplete MCTS panel")
        report[panel] = dict(
            uniform=summarize(old),
            belief=summarize(new),
            wins=paired_comparison(old, new, uniform["manifest"]["seed"]),
            points=paired_points_comparison(old, new, uniform["manifest"]["seed"]),
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uniform")
    parser.add_argument("belief")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = compare(args.uniform, args.belief)
    path = Path(args.out)
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
