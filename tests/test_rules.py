from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from skat.card import Card
from skat.deck import Deck
from skat.game import Round
from skat.hand import FULL_HAND
from skat.models.observation import INPUT_SIZE
from skat.rules import GameDeclaration, evaluate_game, null_overbid_substitute


def score(d=None, points=(61, 59), tricks=(5, 5), bid=0, **kwargs):
    return evaluate_game(
        d or GameDeclaration("suit", 3), 1, points, tricks, bid, **kwargs
    )


@pytest.mark.parametrize(
    "points,won",
    [
        ((60, 60), False),
        ((61, 59), True),
        ((90, 30), True),
        ((30, 90), False),
        ((31, 89), False),
    ],
)
def test_point_boundaries(points, won):
    result = score(points=points)
    assert result.soloist_won is won
    assert result.stages == (2 if min(points) <= 30 else 1)
    assert result.score == result.game_value * (1 if won else -2)


def test_zero_point_trick_is_not_schwarz():
    result = score(points=(120, 0), tricks=(9, 1))
    assert result.stages == 2
    assert score(points=(120, 0), tricks=(10, 0)).stages == 3


@pytest.mark.parametrize(
    "hand,ouvert,value",
    [(False, False, 23), (True, False, 35), (False, True, 46), (True, True, 59)],
)
def test_null_variants(hand, ouvert, value):
    d = GameDeclaration("null", hand=hand, ouvert=ouvert)
    assert score(d, points=(0, 120), tricks=(0, 10)).score == value
    result = score(d, points=(0, 120), tricks=(1, 9))
    assert result.soloist_won is False
    assert result.score == -2 * value


@pytest.mark.parametrize("d", [GameDeclaration("suit", 0), GameDeclaration("grand")])
def test_earned_grade_can_cover_bid(d):
    # With one, Spiel two; a bid for three is covered by earned Schneider.
    bid = 3 * d.base_value
    assert not score(d, points=(90, 30), bid=bid).overbid
    result = score(d, bid=bid)
    assert result.overbid
    assert result.score == -2 * bid


@pytest.mark.parametrize(
    "points,tricks,won,stages",
    [
        ((89, 31), (8, 2), False, 4),
        ((90, 30), (8, 2), True, 4),
        ((120, 0), (10, 0), True, 5),
        ((0, 120), (0, 10), False, 5),
    ],
)
def test_schneider_announcement(points, tricks, won, stages):
    result = score(
        GameDeclaration("grand", hand=True, schneider_announced=True), points, tricks
    )
    assert result.soloist_won is won
    assert result.stages == stages


def test_schwarz_and_ouvert_require_all_tricks():
    for ouvert in (False, True):
        d = GameDeclaration("grand", hand=True, ouvert=ouvert, schwarz_announced=True)
        assert d.schneider_announced
        assert not score(d, points=(120, 0), tricks=(9, 1)).soloist_won
        assert score(d, points=(120, 0), tricks=(10, 0)).stages == 6 + ouvert
    assert (
        evaluate_game(
            GameDeclaration("grand", hand=True, ouvert=True), 4, (120, 0), (10, 0)
        ).score
        == 264
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="grand", ouvert=True),
        dict(kind="suit", suit=0, schneider_announced=True),
        dict(kind="null", schwarz_announced=True),
        dict(kind="suit", suit=4),
        dict(kind="grand", suit=0),
    ],
)
def test_invalid_declarations(kwargs):
    with pytest.raises(ValueError):
        GameDeclaration(**kwargs)


def test_null_overbid_uses_point_game_not_null_multiple():
    cards = [Card(3, 3)] + FULL_HAND[:11]
    value, d = null_overbid_substitute(cards, 24)
    result = evaluate_game(
        GameDeclaration("null"),
        0,
        (0, 120),
        (0, 10),
        24,
        null_overbid_value=value,
        null_overbid_declaration=d,
    )
    assert result.overbid and not result.soloist_won
    assert result.score == -2 * value
    assert result.scored_declaration.kind in ("suit", "grand")
    assert value >= 24


def prepared(d=None, hand=True, seed=1):
    r = Round(
        start=False,
        skip_bidding=True,
        solo_player_id=0,
        hand_game=hand,
        declare_game=d or GameDeclaration("suit", 3),
        seed=seed,
    )
    r.prepare()
    return r


def test_engine_owns_card_removal_and_checks_actions():
    r = prepared()
    seat = r.next_player
    before = tuple(r.player[seat].hand)
    card = r.player[seat].strategy.choose_card(set(before))
    assert tuple(r.player[seat].hand) == before
    with pytest.raises(ValueError):
        r.play_card((seat + 1) % 3, card)
    assert tuple(r.player[seat].hand) == before
    r.play_card(seat, card)
    assert len(r.player[seat].hand) == 9
    with pytest.raises(ValueError):
        r.play_card(seat, card)


def test_follow_suit_violation_leaves_state_unchanged():
    for seed in range(20):
        r = prepared(seed=seed)
        first = r.next_player
        for lead in r.player[first].hand:
            r.game.trick.append(first, lead)
            seat = r.next_player
            legal = r.player[seat].valid_moves(r.game.trick)
            illegal = set(r.player[seat].hand) - legal
            if illegal:
                before = tuple(r.player[seat].hand)
                with pytest.raises(ValueError):
                    r.play_card(seat, next(iter(illegal)))
                assert tuple(r.player[seat].hand) == before
                return
            r.game.new_trick()
    pytest.fail("fixture did not find a follow-suit case")


def test_views_do_not_expose_hidden_skat_or_opponent_cards():
    r = prepared()
    for seat in range(3):
        view = r.view(seat)
        assert view.skat == ()
        assert view.exposed_hands == ((), (), ())
        assert not hasattr(view, "player") and not hasattr(view, "public_state")
        assert view.observation.shape == (INPUT_SIZE,)
        with pytest.raises(FrozenInstanceError):
            view.hand = ()
        with pytest.raises(FrozenInstanceError):
            view.hand[0].rank = "A"
        view.game.trick.append(0, view.hand[0])
        assert len(r.game.trick) == 0


def test_pickup_skat_only_visible_to_soloist():
    r = prepared(hand=False)
    assert len(r.view(0).skat) == 2
    assert not r.view(1).skat
    assert r.get_state(0)[344:].sum() == 2
    assert r.get_state(1)[344:].sum() == 0


def test_ouvert_reveals_cards_before_first_lead():
    r = prepared(GameDeclaration("null", hand=True, ouvert=True))
    assert r.view(1).exposed_hands[0] == tuple(r.player[0].hand)
    assert r.get_state(1)[309:341].sum() == 10  # soloist is previous relative seat
    r.play_card(
        r.next_player, next(iter(r.player[r.next_player].valid_moves(r.game.trick)))
    )
    assert r.view(2).exposed_hands[0] == tuple(r.player[0].hand)


def test_hidden_world_changes_do_not_change_observation():
    r = prepared()
    original = r.get_state(1)
    # Exchange hidden soloist/skat cards. Their points and matadors may differ.
    a, b = r.player[0].hand[0], r.skat[0]
    r.player[0].hand[0], r.skat[0] = b, a
    assert np.array_equal(original, r.get_state(1))


def test_egocentric_seat_rotation():
    r = prepared(hand=False)
    before = [r.get_state(s) for s in range(3)]
    r.player = [r.player[-1], *r.player[:-1]]
    for seat, p in enumerate(r.player):
        p.seat_id = seat
    r.dealer = (r.dealer + 1) % 3
    r.solo_player_id = (r.solo_player_id + 1) % 3
    for seat in range(3):
        assert np.array_equal(before[seat], r.get_state((seat + 1) % 3))


def test_concession_consent_after_second_soloist_card():
    r = prepared()
    while len(r.player[0].hand) >= 9:
        r.step()
    assert not r.concede(0)
    assert not r.is_finished
    with pytest.raises(ValueError):
        r.concede(0, (0,))
    assert r.concede(0, (1,))
    assert r.result.reason == "conceded" and not r.result.soloist_won


def test_early_concession_and_defender_concession():
    r = prepared()
    assert r.concede(0)
    assert r.result.stages == 2  # Hand + Spiel; no artificial Schwarz
    r = prepared()
    assert r.concede(1)
    assert r.result.soloist_won


def test_claim_requires_all_defenders_and_rejection_exposes_cards():
    r = prepared()
    assert not r.claim(0, consenting_defenders=(1,))
    assert not r.is_finished and r.view(2).exposed_hands[0]
    assert r.claim(0, consenting_defenders=(1, 2))
    assert r.result.soloist_won and r.result.reason == "claimed"
    with pytest.raises(ValueError):
        r.play_card(r.next_player, r.player[r.next_player].hand[0])


def test_claim_cannot_ignore_announced_objective():
    r = prepared(GameDeclaration("grand", hand=True, schwarz_announced=True))
    with pytest.raises(ValueError):
        r.claim(0, 0, (1, 2))
    assert r.claim(0, 2, (1, 2))
    assert r.result.stages == 6


@pytest.mark.parametrize(
    "d",
    [
        *(GameDeclaration("suit", s) for s in range(4)),
        GameDeclaration("grand"),
        GameDeclaration("null"),
        GameDeclaration("null", hand=True, ouvert=True),
        GameDeclaration("grand", hand=True, ouvert=True),
    ],
)
def test_full_rounds_preserve_pack_and_public_outcome(d):
    r = prepared(d, hand=d.hand, seed=11)
    while not r.is_finished:
        r.step()
    assert r.cumulative_reward(0) == -r.cumulative_reward(1) == -r.cumulative_reward(2)
    if len(r.trick_history) == 10:
        assert r.points_soloist + r.points_defenders == 120
        played = [card for t in r.trick_history.buffer for _, card in t.buffer]
        assert len(played) == len(set(played)) == 30
        assert set(played + r.skat) == set(FULL_HAND)


def test_fixed_deal_pickup_uses_recorded_skat():
    deal = Deck.factory(seed=8).to_list()
    r = Round(
        start=False,
        skip_bidding=True,
        solo_player_id=0,
        hand_game=False,
        initial_cards=deal,
        declare_game=GameDeclaration("suit", 3),
    )
    r.prepare()
    assert set(r.original_solo_cards) == set(deal[0] + deal[3])
    assert len(r.player[0].hand) == 10 and len(r.skat) == 2


@pytest.mark.parametrize("lengths", [(3, 3, 3, 2), (10, 10, 10, 1), (11, 9, 10, 2)])
def test_invalid_fixed_deals_rejected(lengths):
    blocks = []
    i = 0
    for n in lengths:
        blocks.append(FULL_HAND[i : i + n])
        i += n
    with pytest.raises(ValueError):
        Round(start=False, initial_cards=blocks)


def test_concession_cannot_reverse_secured_result():
    d = GameDeclaration("suit", 3)
    assert score(
        d, points=(61, 30), tricks=(5, 3), forced_winner=False, awarded_level=0
    ).soloist_won
    assert not score(
        d, points=(20, 60), tricks=(3, 5), forced_winner=True, awarded_level=0
    ).soloist_won
    d = GameDeclaration("grand", hand=True, schwarz_announced=True)
    assert not score(
        d, points=(100, 0), tricks=(8, 1), forced_winner=True, awarded_level=2
    ).soloist_won


def test_preset_hand_declaration_does_not_pick_up_skat():
    r = Round(
        start=False,
        skip_bidding=True,
        solo_player_id=0,
        declare_game=GameDeclaration("null", hand=True),
        seed=1,
    )
    r.prepare()
    assert r.hand_game is True and not r.view(0).skat


@pytest.mark.parametrize("can_win", [False, True])
def test_defender_opening_checks_all_legal_continuations(can_win):
    from skat.rules import can_soloist_take_trick

    for seed in range(30):
        r = prepared(seed=seed)
        while len(r.trick_history) < 9:
            r.step()
        possible = can_soloist_take_trick(
            r.game, [list(p.hand) for p in r.player], r.next_player, 0
        )
        if possible is can_win:
            before_solo = r.party_tricks[0]
            r.open_defender(1)
            assert r.is_finished
            assert r.party_tricks == (
                before_solo + int(can_win),
                10 - before_solo - int(can_win),
            )
            assert r.points_soloist + r.points_defenders == 120
            assert r.result.reason == "defender-open"
            return
    pytest.fail("no matching endgame fixture found")


def test_defender_opening_takes_current_table_cards_into_account():
    r = prepared(seed=4)
    while len(r.trick_history) < 9:
        r.step()
    r.step()
    assert len(r.game.trick) == 1
    r.open_defender(2)
    assert sum(r.party_tricks) == 10
    assert r.points_soloist + r.points_defenders == 120


def test_defender_concession_awards_required_possible_grade():
    r = prepared(GameDeclaration("suit", 3), hand=False)
    matadors = r.game.matadors(r.original_solo_cards)
    r.highest_bid = (matadors + 2) * 12
    r.concede(1)
    assert r.result.soloist_won and not r.result.overbid
    assert r.result.stages == 2


def test_defender_concession_cannot_award_already_impossible_schwarz():
    r = prepared(GameDeclaration("suit", 3), hand=False)
    while r.party_tricks[1] == 0:
        r.step()
        if r.is_finished:
            pytest.fail("fixture should include a defender trick")
    r.highest_bid = (r.game.matadors(r.original_solo_cards) + 3) * 12
    r.concede(1)
    assert not r.result.soloist_won and r.result.overbid


@pytest.mark.parametrize("suit", [True, 0.0, "0"])
def test_declaration_rejects_noninteger_suit(suit):
    with pytest.raises(ValueError):
        GameDeclaration("suit", suit=suit)


def test_declaration_rejects_nonboolean_flags():
    with pytest.raises(ValueError):
        GameDeclaration("grand", hand=1)


def test_replayed_pressed_hand_preserves_skat_knowledge():
    r = Round(
        solo_player_id=0,
        skip_bidding=True,
        declare_game=GameDeclaration("grand"),
        hand_game=True,
        replay=True,
        seed=7,
        start=False,
    )
    r.prepare()
    assert len(r.player[0].hand) == 10
    assert r.view(0).hand_game is False
    assert set(r.view(0).skat) == set(r.skat)
    assert not r.view(1).skat
    assert r.get_state(0)[344:376].sum() == 2
