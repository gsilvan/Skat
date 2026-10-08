"""Changing hidden cards must not change observations or a seeded search."""

from random import Random
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch

from skat.agents.belief_recorder import belief_input
from skat.agents.dqn import DQNAgent
from skat.agents.mcts import MCTSAgent
from skat.agents.random import RandomAgent
from skat.agents.rl.result_q import resolved_result, secured_victory
from skat.card import Card
from skat.game import GamePhase, Round
from skat.games.grand import Grand, GrandTrick
from skat.hand import FULL_HAND, Hand


def card(suit, rank):
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def position(rich_skat, *, observer=1, hand_game=False):
    agent = MCTSAgent(seed=7, simulations=60, determinizations=3)
    agents = [RandomAgent(seed=0), RandomAgent(seed=1), RandomAgent(seed=2)]
    agents[observer] = agent
    round_ = Round(
        start=False,
        solo_player_id=0,
        hand_game=hand_game,
        declare_game=Grand(),
        agents=agents,
        phase=GamePhase.PLAYING,
    )
    played = []
    for suit in ("♥", "♠"):
        trick = GrandTrick()
        for seat, rank in enumerate(("A", "X", "K")):
            c = card(suit, rank)
            trick.append(seat, c)
            played.append(c)
        assert trick.winner == 0
        round_.trick_history.buffer.append(trick)
    high = [card("♦", "A"), card("♦", "X")]
    low = [card("♦", "7"), card("♦", "8")]
    round_.skat = high if rich_skat else low
    fixed_hand = [c for c in FULL_HAND if c not in played + high + low][:8]
    solo_hand = [c for c in FULL_HAND if c not in played + high + low + fixed_hand][:8]
    remaining = [
        c for c in FULL_HAND if c not in played + fixed_hand + solo_hand + round_.skat
    ]
    round_.player[1].hand = Hand(fixed_hand)
    round_.player[0].hand = Hand(solo_hand)
    round_.player[2].hand = Hand(remaining)
    if observer == 2:
        round_.player[1].hand, round_.player[2].hand = (
            round_.player[2].hand,
            round_.player[1].hand,
        )
    round_.player[0].trick_stack = Hand([*played, *round_.skat])
    round_.game_value = round_.game.spielwert(
        [*round_.player[0].hand, card("♥", "A"), card("♠", "A"), *round_.skat]
    )
    return round_, agent


@pytest.mark.parametrize("hand_game", [False, True])
@pytest.mark.parametrize("observer", [1, 2])
def test_defender_observation_and_belief_ignore_true_skat(hand_game, observer):
    a, _ = position(True, hand_game=hand_game, observer=observer)
    b, _ = position(False, hand_game=hand_game, observer=observer)
    assert a.points_soloist == 71 and b.points_soloist == 50
    assert a.observer_points(observer) == b.observer_points(observer) == (0, 50)
    np.testing.assert_array_equal(a.get_state(observer), b.get_state(observer))
    np.testing.assert_array_equal(belief_input(a, observer), belief_input(b, observer))
    assert (
        a.observer_victory_status(observer) == b.observer_victory_status(observer) == 0
    )
    assert not resolved_result(a.get_state_t(observer)).item()


@pytest.mark.parametrize("hand_game,expected", [(False, 71), (True, 50)])
def test_soloist_only_counts_skat_after_pickup(hand_game, expected):
    r, _ = position(True, hand_game=hand_game)
    assert r.observer_points(0) == (expected, 0)
    assert secured_victory(r.get_state_t(0)).item() == (not hand_game)
    assert r.points_soloist == 71  # actual scoring still includes every skat


@pytest.mark.parametrize("hand_game", [False, True])
@pytest.mark.parametrize("observer", [1, 2])
def test_seeded_defender_search_is_identical_in_indistinguishable_worlds(
    hand_game, observer
):
    a, agent_a = position(True, hand_game=hand_game, observer=observer)
    _b, agent_b = position(False, hand_game=hand_game, observer=observer)
    root_a, root_b = agent_a.determinize(), agent_b.determinize()
    assert root_a.hands == root_b.hands
    assert root_a.points == root_b.points
    assert root_a.secured_winner == root_b.secured_winner == 0
    assert root_a.overbid == root_b.overbid
    agent_a.rng = Random(10)
    agent_b.rng = Random(10)
    moves = sorted(a.player[observer].hand, key=lambda c: c.np_index)
    totals_a, totals_b = agent_a.tally(moves), agent_b.tally(moves)
    assert totals_a == totals_b
    assert totals_a.bonus_sums == totals_b.bonus_sums
    assert agent_a.best(moves, totals_a) == agent_b.best(moves, totals_b)


def test_unknown_hand_skat_does_not_enter_soloist_observation():
    a, _ = position(True, observer=0, hand_game=True)
    b, _ = position(False, observer=0, hand_game=True)
    np.testing.assert_array_equal(a.get_state(0), b.get_state(0))
    assert a.observer_victory_status(0) == b.observer_victory_status(0) == 0


def test_mcts_does_not_read_omniscient_overbid():
    r, agent = position(True)
    r.highest_bid = 60  # hidden matadors can determine whether this is affordable
    assert r.observer_overbid(1) is None
    with patch.object(
        Round,
        "overbid",
        new_callable=PropertyMock,
        side_effect=AssertionError("hidden engine value read"),
    ):
        root = agent.determinize()
    cards = [*root.hands[0], card("♥", "A"), card("♠", "A")]
    unseen_skat = set(FULL_HAND) - set(cards)
    unseen_skat -= set(root.hands[1]) | set(root.hands[2])
    unseen_skat -= {card(s, rank) for s in ("♥", "♠") for rank in ("X", "K")}
    assert root.overbid == (r.highest_bid > r.game.spielwert([*cards, *unseen_skat]))


def test_known_public_defence_victory_still_unlocks_bonus():
    r, _ = position(True)
    r.trick_history.buffer.clear()
    for suit, ranks in (
        ("♥", ("X", "A", "K")),
        ("♠", ("X", "A", "K")),
        ("♣", ("7", "X", "8")),
    ):
        trick = GrandTrick()
        for seat, rank in enumerate(ranks):
            c = card(suit, rank)
            trick.append(seat, c)
            for player in r.player:
                if c in player.hand:
                    player.hand.remove(c)
        assert trick.winner == 1
        r.trick_history.buffer.append(trick)
    assert r.observer_points(1) == (60, 0)
    assert r.observer_victory_status(1) == -1
    assert secured_victory(r.get_state_t(1)).item()


@pytest.mark.parametrize("observer,expected", [(0, False), (1, True), (2, True)])
def test_dqn_bonus_permission_uses_observer_overbid_not_hidden_truth(
    observer, expected
):
    from unittest.mock import Mock

    r, _ = position(True, observer=observer, hand_game=True)
    r.highest_bid = 60
    assert r.observer_overbid(observer) is None
    agent = DQNAgent.__new__(DQNAgent)
    agent.state = r.player[observer]
    agent.train = False
    moves = set(agent.state.hand)
    choice = min(moves, key=lambda c: c.np_index)
    prediction = torch.zeros(32)
    prediction[choice.np_index] = 1
    agent.dqn = Mock()
    agent.dqn.select_action.return_value = prediction
    assert agent.choose_card(moves) == choice
    assert agent.dqn.select_action.call_args.kwargs["allow_bonus"] is expected


@pytest.mark.parametrize("overbid,expected", [(None, 0), (False, 1), (True, -1)])
def test_public_soloist_points_do_not_override_uncertain_overbidding(overbid, expected):
    r, _ = position(True)
    with (
        patch.object(r, "observer_points", return_value=(0, 61)),
        patch.object(r, "observer_overbid", return_value=overbid),
    ):
        assert r.observer_victory_status(1) == expected
