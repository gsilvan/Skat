from random import Random

import torch

from skat.agents.distill.targets import value_targets
from skat.agents.mcts import MCTSAgent, MoveTotals
from skat.agents.search.uct import Node
from skat.hand import FULL_HAND


def test_root_bonus_never_compensates_for_lower_win_rate():
    moves = list(FULL_HAND[:2])
    totals = MoveTotals(moves)
    totals[moves[0]] = [1_000_000, 500_001.0]
    totals[moves[1]] = [1_000_000, 500_000.0]
    totals.bonus_sums[moves[1]] = 1_000_000
    assert MCTSAgent.best(moves, totals) == moves[0]


def test_root_exact_tie_uses_mean_bonus_not_total_bonus_or_visits():
    moves = list(FULL_HAND[:2])
    totals = MoveTotals(moves)
    totals[moves[0]], totals[moves[1]] = [30, 10.0], [3, 1.0]
    totals.bonus_sums[moves[0]], totals.bonus_sums[moves[1]] = 9.0, 2.0
    assert MCTSAgent.best(moves, totals) == moves[1]


def test_tree_only_breaks_exact_ucb_ties():
    root = Node(0, [])
    a, b = Node(1, []), Node(1, [])
    root.children, root.visits = [a, b], 20
    a.visits = b.visits = 10
    a.value_sum = b.value_sum = 5.0
    a.bonus_sum, b.bonus_sum = 0.0, 10.0
    assert root.select_child(1.4, bonus_tiebreak=True) is b
    a.value_sum += 0.0000001
    assert root.select_child(1.4, bonus_tiebreak=True) is a


def test_distillation_resolves_ties_without_boosting_lower_win_rates():
    visits = torch.tensor([[10.0, 20.0, 10.0, 0.0]])
    values = torch.tensor([[5.0, 10.0, 4.0, 0.0]])
    bonuses = torch.tensor([[1.0, 15.0, 1_000_000.0, 1_000_000.0]])
    mask = torch.tensor([[True, True, True, False]])
    target = value_targets(values, visits, mask, bonuses=bonuses)
    assert target.argmax().item() == 1
    assert target[0, 1] > target[0, 0] > target[0, 2]
    assert target[0, 3] == 0


def test_bidding_outcome_only_does_not_use_bonus(monkeypatch):
    from skat.agents.search.playout import PlayoutState
    from skat.agents.search.uct import uct_search
    from tests.agents.search.test_uct import UctSearchTest

    def forbidden(self):
        raise AssertionError("bidding must remain probability-only")

    monkeypatch.setattr(PlayoutState, "bonus_rewards", forbidden)
    result = uct_search(
        UctSearchTest().endgame(), 40, Random(0), outcome_only=True, bonus_tiebreak=True
    )
    assert all(value == 0 for value in result.bonus_sums.values())
