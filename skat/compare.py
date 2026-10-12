"""Compare complete evaluations using identical deals and fixed references."""

import argparse
import json
from pathlib import Path

from skat.evaluation import paired_comparison, paired_points_comparison, summarize


def compare(first, second, output=None):
    a, b = (json.loads(Path(path).read_text()) for path in (first, second))
    for key in (
        "rules_schema",
        "observation_schema",
        "source_sha256",
        "seed",
        "deals",
        "deal_offset",
        "opponent_policy_sha256",
        "mcts",
        "mcts_rng_scheme",
    ):
        if a["manifest"].get(key) != b["manifest"].get(key):
            raise ValueError(
                f"evaluations differ in {key}; regenerate with fixed references"
            )
    candidates = [r for r in b["outcomes"] if r["agent"] == "policy"]
    if not candidates:
        raise ValueError("second evaluation must contain a policy candidate")
    expected = 6 * a["manifest"]["deals"]

    def check(rows):
        keys = {(r["panel"], r["seat"], r["seed"]) for r in rows}
        if len(rows) != expected or len(keys) != expected:
            raise ValueError("incomplete or duplicate evaluation outcomes")
        return keys

    keys = check(candidates)
    comparisons = {}
    for agent in a["manifest"]["agents"]:
        baseline = [r for r in a["outcomes"] if r["agent"] == agent]
        if check(baseline) != keys:
            raise ValueError("evaluation deals/seats do not match")
        panels = {}
        for panel in ("random", "policy"):
            old = [r for r in baseline if r["panel"] == panel]
            new = [r for r in candidates if r["panel"] == panel]
            panels[panel] = dict(
                baseline=summarize(old),
                candidate=summarize(new),
                paired=paired_comparison(old, new, a["manifest"]["seed"]),
                points_paired=paired_points_comparison(old, new, a["manifest"]["seed"]),
            )
        comparisons[agent] = panels
    by_game = {}
    for panel in ("random", "policy"):
        by_game[panel] = {
            kind: summarize(
                [
                    r
                    for r in candidates
                    if r["panel"] == panel
                    and (r["result"].get("declaration") or {}).get("kind") == kind
                ]
            )
            for kind in ("suit", "grand", "null")
        }
    report = dict(
        baseline=str(first),
        candidate=str(second),
        comparisons=comparisons,
        candidate_by_game=by_game,
    )
    if output:
        path = Path(output)
        with path.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(args.baseline, args.candidate, args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for agent, panels in report["comparisons"].items():
        for panel, values in panels.items():
            print(f"{agent} -> ISS against {panel}: {values['paired']}")


if __name__ == "__main__":
    main()
