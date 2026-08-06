"""
`CardTransformer` is the second reader of the 240-wide observation, and unlike
`PolicyNet` it does not merely consume it - it takes the layout apart by index.
`Round.get_state` and those indices are coupled by hand, exactly as
`QNet.INPUT_SIZE` is, and this file is the only thing that says so.

Every case here starts from a real `Round`. A synthetic 240-vector would pin
the tokeniser against my reading of the layout rather than against the layout,
which is the one thing it exists to check.

The block offsets are restated locally for the same reason `test_state_vector`
restates them: importing `CardTransformer.STATUS_BLOCKS` would make this file
agree with the tokeniser by construction and pin nothing at all.
"""

import random
import unittest

import numpy as np
import torch

from skat.agents.random import RandomAgent
from skat.card import Card
from skat.game import GamePhase, Round
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND
from skat.models.card_transformer import CardTransformer

# The seven columns, in the order the tokeniser is expected to produce them.
HELD, PLAYED_ME, PLAYED_NEXT, PLAYED_PREV = 0, 1, 2, 3
CURRENT_ME, CURRENT_NEXT, CURRENT_PREV = 4, 5, 6
UNSEEN = 7
IS_TRUMP = 8
ORDER = slice(9, 12)
SAME_SUIT = 12

CARDS = 32


def make_round(hands, soloist=0, game=None, steps=0, dealer=0) -> Round:
    """`hands` is [seat0, seat1, seat2, skat]."""
    round_ = Round(
        dealer=dealer,
        skip_bidding=True,
        solo_player_id=soloist,
        hand_game=True,
        declare_game=game if game is not None else SuitGame(3),
        agents=[RandomAgent() for _ in range(3)],
        start=False,
        initial_cards=[list(h) for h in hands],
    )
    round_.deal()
    round_.phase = GamePhase.PLAYING
    for _ in range(steps):
        round_.step()
    return round_


DEAL = [FULL_HAND[0:10], FULL_HAND[10:20], FULL_HAND[20:30], FULL_HAND[30:32]]

CLUBS = 3


def tokens_of(round_: Round, seat: int = 0, net: CardTransformer = None) -> np.ndarray:
    """The `(32, F)` token table the network builds for `seat`."""
    net = net or CardTransformer()
    state = torch.tensor(round_.get_state(player_id=seat)).unsqueeze(0)
    with torch.no_grad():
        return net.tokens(state)[0].numpy()


def slots(column: np.ndarray) -> set:
    """Which card slots a column is set for."""
    return set(np.nonzero(column)[0].tolist())


def held_slots(round_: Round, seat: int) -> set:
    return {card.np_index for card in round_.player[seat].hand}


class StatusColumnTest(unittest.TestCase):
    def test_the_tokeniser_puts_my_hand_in_the_right_column(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=6)
        for seat in range(3):
            table = tokens_of(round_, seat)
            self.assertEqual(held_slots(round_, seat), slots(table[:, HELD]))

    def test_a_played_card_moves_out_of_the_hand_into_a_history_column(self) -> None:
        before = tokens_of(make_round(DEAL, soloist=0, steps=0), seat=0)
        after = tokens_of(make_round(DEAL, soloist=0, steps=6), seat=0)
        gone = slots(before[:, HELD]) - slots(after[:, HELD])
        self.assertEqual(2, len(gone))  # two completed tricks
        self.assertEqual(gone, slots(after[:, PLAYED_ME]))

    def test_the_three_history_columns_are_egocentric(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=6)
        for seat in range(3):
            table = tokens_of(round_, seat)
            for offset, column in enumerate((PLAYED_ME, PLAYED_NEXT, PLAYED_PREV)):
                expected = round_.trick_history.to_numpy(player_id=(seat + offset) % 3)
                np.testing.assert_array_equal(expected, table[:, column])

    def test_the_live_trick_lands_in_current_and_not_in_played(self) -> None:
        # seven steps: two complete tricks plus one card on the table
        round_ = make_round(DEAL, soloist=0, steps=7)
        assert round_.game is not None
        table = tokens_of(round_, seat=0)
        live = {card.np_index for _, card in round_.game.trick.buffer}
        self.assertEqual(1, len(live))
        current = slots(
            table[:, CURRENT_ME] + table[:, CURRENT_NEXT] + table[:, CURRENT_PREV]
        )
        played = slots(
            table[:, PLAYED_ME] + table[:, PLAYED_NEXT] + table[:, PLAYED_PREV]
        )
        self.assertEqual(live, current)
        self.assertEqual(set(), live & played)

    def test_every_card_is_in_exactly_one_status_column_or_unseen(self) -> None:
        for steps in (0, 5, 7, 14, 22, 29):
            table = tokens_of(make_round(DEAL, soloist=0, steps=steps))
            status = table[:, HELD:UNSEEN]
            np.testing.assert_array_equal(
                np.ones(CARDS), status.sum(axis=1) + table[:, UNSEEN]
            )
            self.assertLessEqual(status.sum(axis=1).max(), 1.0)

    def test_unseen_counts_the_skat_and_the_hands_i_cannot_see(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=7)
        table = tokens_of(round_, seat=0)
        hidden = held_slots(round_, 1) | held_slots(round_, 2)
        hidden |= {card.np_index for card in round_.skat}
        self.assertEqual(hidden, slots(table[:, UNSEEN]))


class TrumpTest(unittest.TestCase):
    """`is_trump` is derived from the 6-wide game block, and it leans on
    `Game.to_numpy` setting a suit bit only when the game has a trump suit.
    Nothing else in the tree asserts that, so it is asserted here."""

    def trumps(self, game) -> set:
        return slots(tokens_of(make_round(DEAL, soloist=0, game=game))[:, IS_TRUMP])

    def test_the_jacks_are_trumps_in_every_suit_game(self) -> None:
        jacks = {card.np_index for card in FULL_HAND if card.is_jack}
        for suit in range(4):
            self.assertTrue(jacks <= self.trumps(SuitGame(suit)))

    def test_the_trump_suit_follows_the_declaration(self) -> None:
        for suit in range(4):
            expected = {
                c.np_index for c in FULL_HAND if c.is_jack or c.suit == "♦♥♠♣"[suit]
            }
            self.assertEqual(expected, self.trumps(SuitGame(suit)))

    def test_only_the_jacks_are_trumps_in_a_grand(self) -> None:
        self.assertEqual(
            {card.np_index for card in FULL_HAND if card.is_jack}, self.trumps(Grand())
        )

    def test_nothing_is_trump_in_a_null(self) -> None:
        self.assertEqual(set(), self.trumps(Null()))

    def test_the_trump_jack_is_not_counted_twice(self) -> None:
        # the clamp: in a clubs game ♣J is both a jack and of the trump suit
        table = tokens_of(make_round(DEAL, soloist=0, game=SuitGame(CLUBS)))
        self.assertEqual(1.0, table[Card(CLUBS, 3).np_index, IS_TRUMP])


class TrickOrderTest(unittest.TestCase):
    """Which card led the live trick decides who is currently winning it. The
    observation attributes the trick by seat but does not order it - the order
    is derived, and this is where that derivation is checked against the trick
    the engine actually holds."""

    def played_order(self, round_: Round) -> dict:
        assert round_.game is not None
        return {
            card.np_index: i for i, (_, card) in enumerate(round_.game.trick.buffer)
        }

    def derived_order(self, table: np.ndarray) -> dict:
        order = table[:, ORDER]
        return {
            slot: int(order[slot].argmax())
            for slot in range(CARDS)
            if order[slot].any()
        }

    def test_the_trick_order_matches_the_order_the_cards_were_played(self) -> None:
        for steps in range(6, 30):
            round_ = make_round(DEAL, soloist=0, steps=steps)
            table = tokens_of(round_, seat=0)
            self.assertEqual(
                self.played_order(round_),
                self.derived_order(table),
                f"after {steps} steps",
            )

    def test_an_empty_trick_has_no_order_at_all(self) -> None:
        table = tokens_of(make_round(DEAL, soloist=0, steps=6))
        self.assertEqual({}, self.derived_order(table))
        np.testing.assert_array_equal(np.zeros((CARDS, 3)), table[:, ORDER])

    def test_the_order_is_right_from_every_chair(self) -> None:
        # the derivation runs off FRONT_HAND, which differs per observer, so
        # all three must recover the same absolute play order
        round_ = make_round(DEAL, soloist=0, steps=8)
        expected = self.played_order(round_)
        for seat in range(3):
            self.assertEqual(expected, self.derived_order(tokens_of(round_, seat)))

    def test_the_led_suit_is_the_suit_of_the_card_that_led(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=8)
        assert round_.game is not None
        led = round_.game.trick.buffer[0][1]
        table = tokens_of(round_, seat=0)
        expected = {c.np_index for c in FULL_HAND if c.suit == led.suit}
        self.assertEqual(expected, slots(table[:, SAME_SUIT]))

    def test_nothing_shares_a_suit_with_a_card_that_was_never_led(self) -> None:
        table = tokens_of(make_round(DEAL, soloist=0, steps=6))
        self.assertEqual(set(), slots(table[:, SAME_SUIT]))


class SeatInvarianceTest(unittest.TestCase):
    """The observation is seat-invariant; the tokens must inherit that, or the
    one-checkpoint-for-three-seats premise stops at the network boundary."""

    def rotated(self, hands, soloist, dealer) -> Round:
        # `RandomAgent` draws from the global RNG, so without re-seeding the
        # three rotated rounds play different cards and the comparison would be
        # between different positions rather than between different chairs.
        random.seed(11)
        return make_round(hands, soloist=soloist, dealer=dealer, steps=7)

    def test_a_relabelled_deal_gives_the_same_token_table(self) -> None:
        a, b, c, skat = DEAL
        base = self.rotated([a, b, c, skat], 0, 0)
        by_one = self.rotated([c, a, b, skat], 1, 1)
        by_two = self.rotated([b, c, a, skat], 2, 2)

        first = tokens_of(base, 0)
        np.testing.assert_array_equal(first, tokens_of(by_one, 1))
        np.testing.assert_array_equal(first, tokens_of(by_two, 2))

    def test_it_still_distinguishes_genuinely_different_seats(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=7)
        self.assertFalse(
            np.array_equal(tokens_of(round_, 0), tokens_of(round_, 1)),
            "different hands and roles must not collapse to the same tokens",
        )


class RawFeaturesTest(unittest.TestCase):
    def test_the_control_keeps_the_status_columns_and_drops_the_rest(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=7)
        full = tokens_of(round_, 0)
        raw = tokens_of(round_, 0, net=CardTransformer(features="raw"))
        self.assertEqual((CARDS, 8), raw.shape)
        np.testing.assert_array_equal(full[:, : UNSEEN + 1], raw)


class ContextTest(unittest.TestCase):
    def test_the_context_is_the_global_scalars_plus_led_is_trump(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=8, game=SuitGame(CLUBS))
        state = torch.tensor(round_.get_state(player_id=0)).unsqueeze(0)
        net = CardTransformer()
        with torch.no_grad():
            context = net.context_vector(state)[0].numpy()
        self.assertEqual(17, len(context))
        np.testing.assert_array_equal(round_.get_state(0)[32:44], context[:12])
        np.testing.assert_array_equal(round_.get_state(0)[236:240], context[12:16])

    def test_led_is_trump_says_whether_the_led_card_was_a_trump(self) -> None:
        net = CardTransformer()
        for steps in range(6, 30):
            round_ = make_round(DEAL, soloist=0, steps=steps, game=SuitGame(CLUBS))
            assert round_.game is not None
            state = torch.tensor(round_.get_state(player_id=0)).unsqueeze(0)
            with torch.no_grad():
                led_is_trump = float(net.context_vector(state)[0][-1])
            buffer = round_.game.trick.buffer
            trumps = round_.game.trump_cards(round_.game.suit)
            expected = float(buffer[0][1] in trumps) if buffer else 0.0
            self.assertEqual(expected, led_is_trump, f"after {steps} steps")


if __name__ == "__main__":
    unittest.main()
