from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from skat.agents import Agent
from skat.agents.random import RandomAgent
from skat.game import SeatScore, Tournament

AgentFactory = Callable[[int, int], Agent]


def _seed(root: int, seat: int) -> int:
    # Arithmetic, not hash(): Python deliberately salts string hashes per process.
    return (root + 0x9E3779B9 * (seat + 1)) % (2**32)


def _wilson(wins: int, rounds: int, z: float = 1.959963984540054) -> list[float]:
    """95% Wilson interval for a Bernoulli win rate."""
    if not rounds:
        return [0.0, 0.0]
    p = wins / rounds
    denominator = 1 + z * z / rounds
    centre = (p + z * z / (2 * rounds)) / denominator
    spread = (
        z
        * math.sqrt(p * (1 - p) / rounds + z * z / (4 * rounds * rounds))
        / denominator
    )
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def _score_dict(score: SeatScore) -> dict[str, int]:
    return {
        "rounds": score.rounds,
        "wins": score.wins,
        "solo_rounds": score.solo_rounds,
        "solo_wins": score.solo_wins,
        "defence_rounds": score.defence_rounds,
        "defence_wins": score.defence_wins,
        "overbids": score.overbids,
        "card_points": score.card_points,
    }


def evaluate_candidate(
    name: str,
    factory: AgentFactory,
    rounds: int,
    seed: int,
    opponents: dict[str, AgentFactory] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate on every seat against fixed opponent panels.

    Each seat run reuses the same deal seed.  The candidate therefore sees each
    of the three hands from every deal exactly once in every panel. Ranking one
    designated seat avoids double-crediting the two defenders for the same party
    result.  The default single random panel keeps this helper useful in tests;
    the full benchmark adds a competent policy panel because learned beliefs are
    population-specific and are known not to help against random play.
    """
    if opponents is None:
        opponents = {
            "random": lambda opponent_seed, _seat: RandomAgent(seed=opponent_seed)
        }
    totals = {
        "rounds": 0,
        "wins": 0,
        "solo_rounds": 0,
        "solo_wins": 0,
        "defence_rounds": 0,
        "defence_wins": 0,
        "overbids": 0,
        "card_points": 0,
    }
    by_seat = []
    by_opponent = []
    passed = 0
    for opponent_name, opponent_factory in opponents.items():
        opponent_totals = {key: 0 for key in totals}
        opponent_passed = 0
        for candidate_seat in range(3):
            agents: list[Agent] = [
                opponent_factory(_seed(seed, seat), seat) for seat in range(3)
            ]
            candidate = factory(_seed(seed, candidate_seat), candidate_seat)
            agents[candidate_seat] = candidate
            tournament = Tournament(rounds=rounds, agents=agents, seed=seed)
            tournament.start()
            score = _score_dict(tournament.seat_scores[candidate_seat])
            for key in totals:
                totals[key] += score[key]
                opponent_totals[key] += score[key]
            passed += tournament.passed
            opponent_passed += tournament.passed
            by_seat.append(
                {
                    "opponent": opponent_name,
                    "seat": candidate_seat,
                    **score,
                    "passed": tournament.passed,
                    "outcomes": [
                        (
                            None
                            if record["seat_wins"] is None
                            else record["seat_wins"][candidate_seat]
                        )
                        for record in tournament.round_records
                    ],
                    "deals": tournament.round_records,
                }
            )
            if hasattr(candidate, "close"):
                candidate.close()  # type: ignore[attr-defined]
        opponent_rounds = opponent_totals["rounds"]
        by_opponent.append(
            {
                "opponent": opponent_name,
                **opponent_totals,
                "passed": opponent_passed,
                "win_rate": (
                    opponent_totals["wins"] / opponent_rounds
                    if opponent_rounds
                    else 0.0
                ),
            }
        )

    played = totals["rounds"]
    totals.update(
        {
            "name": name,
            "scheduled_rounds": rounds * 3 * len(opponents),
            "passed": passed,
            "win_rate": totals["wins"] / played if played else 0.0,
            "win_rate_95": _wilson(totals["wins"], played),
            "solo_win_rate": (
                totals["solo_wins"] / totals["solo_rounds"]
                if totals["solo_rounds"]
                else 0.0
            ),
            "defence_win_rate": (
                totals["defence_wins"] / totals["defence_rounds"]
                if totals["defence_rounds"]
                else 0.0
            ),
            "card_points_per_round": (
                totals["card_points"] / played if played else 0.0
            ),
            "by_seat": by_seat,
            "by_opponent": by_opponent,
        }
    )
    return totals


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        results,
        key=lambda row: (row["win_rate"], row["card_points_per_round"], row["name"]),
        reverse=True,
    )
    return [{**row, "rank": rank} for rank, row in enumerate(ranked, start=1)]
