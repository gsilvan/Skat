"""Party outcome features, independent of agents and search implementation.

The existing engine thresholds are deliberately retained: soloist 61/91,
defenders 60/90, and the engine's 120-point definition of Schwarz. Primary
win/loss features never depend on bonuses. Points are a separate bonus used
after secured victory, or by MCTS to break exact primary-value ties.
"""

REWARD_OBJECTIVE = "win-first-secured-bonus-v1"
MCTS_TIEBREAK_OBJECTIVE = "win-first-exact-tie-bonus-v1"


def victory_status(
    soloist_points: int, defender_points: int, overbid: bool = False
) -> int:
    """0 while undecided, +1 once solo victory is secure, -1 for defence."""
    if overbid or defender_points >= 60:
        return -1
    return int(soloist_points > 60)


def victory_bonus(points: int, is_soloist: bool) -> float:
    """Nonnegative bonus for a won outcome, separate from win/loss utility.

    Missing an upgrade never incurs a penalty. Stage bonuses favour crossing
    the next threshold; points also give a useful goal when it is unreachable.
    """
    schneider = points > 90 if is_soloist else points >= 90
    return 0.6 * points / 120 + 0.2 * schneider + 0.2 * (points == 120)


def outcome_features(
    soloist_points: int,
    defender_points: int,
    *,
    is_null: bool = False,
    soloist_took_trick: bool = False,
    overbid: bool = False,
) -> tuple[float, float, float]:
    """Signed win, Schneider and Schwarz features from the soloist's view.

    Call with final scores. Null is purely a no-trick outcome and has no
    Schneider/Schwarz upgrades. An overbid loses irrespective of card points.
    """
    if is_null:
        return (-1.0 if overbid or soloist_took_trick else 1.0, 0.0, 0.0)
    won = not overbid and soloist_points > 60
    if won:
        return (1.0, float(soloist_points > 90), float(soloist_points == 120))
    return (-1.0, -float(defender_points >= 90), -float(defender_points == 120))
