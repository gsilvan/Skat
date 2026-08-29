from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _agents(run: Path) -> dict[str, dict[str, Any]]:
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    return {row["name"]: row for row in data["ranking"]}


def _outcomes(row: dict[str, Any]) -> dict[tuple[str, int, int], bool]:
    outcomes = {}
    for block in row["by_seat"]:
        for deal, won in zip(block["deals"], block["outcomes"]):
            if won is not None:
                outcomes[(block["opponent"], block["seat"], deal["seed"])] = won
    return outcomes


def mcnemar_exact(improved: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value over the discordant paired deals."""
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = min(improved, regressed)
    log_two = math.log(2.0)
    probability = sum(
        math.exp(
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * log_two
        )
        for k in range(tail + 1)
    )
    return min(1.0, 2.0 * probability)


def compare_runs(old: Path, new: Path) -> list[dict[str, Any]]:
    """Pair common agents on identical panel/seat/deal seeds."""
    old_agents, new_agents = _agents(old), _agents(new)
    comparison = []
    for name in sorted(old_agents.keys() & new_agents.keys()):
        old_outcomes = _outcomes(old_agents[name])
        new_outcomes = _outcomes(new_agents[name])
        keys = sorted(old_outcomes.keys() & new_outcomes.keys())
        if not keys:
            continue
        old_wins = sum(old_outcomes[key] for key in keys)
        new_wins = sum(new_outcomes[key] for key in keys)
        improved = sum(not old_outcomes[key] and new_outcomes[key] for key in keys)
        regressed = sum(old_outcomes[key] and not new_outcomes[key] for key in keys)
        comparison.append(
            {
                "name": name,
                "paired_deals": len(keys),
                "old_win_rate": old_wins / len(keys),
                "new_win_rate": new_wins / len(keys),
                "delta": (new_wins - old_wins) / len(keys),
                "improved": improved,
                "regressed": regressed,
                "mcnemar_p": mcnemar_exact(improved, regressed),
            }
        )
    return sorted(comparison, key=lambda row: (row["delta"], row["name"]), reverse=True)


def comparison_markdown(old: Path, new: Path) -> str:
    rows = compare_runs(old, new)
    if not rows:
        return (
            "No common agent/deal pairs; use the same benchmark seed to compare runs."
        )
    lines = [
        f"# Benchmark comparison: {old} → {new}",
        "",
        "| agent | old | new | delta | improved/regressed | McNemar p | paired |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['old_win_rate']:.1%} | "
            f"{row['new_win_rate']:.1%} | {row['delta']:+.1%} | "
            f"{row['improved']}/{row['regressed']} | {row['mcnemar_p']:.4g} | "
            f"{row['paired_deals']} |"
        )
    return "\n".join(lines)
