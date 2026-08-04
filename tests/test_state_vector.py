"""
The observation fed to the RL agents is *egocentric*: slot 0 is always the
observer.

That is what makes one network enough. Indexed by absolute seat instead, the
same situation produces a different vector from each chair and the net has to
learn every concept three times - which is the only reason the CLI ever carried
a checkpoint per seat.
"""

import unittest

import numpy as np

from skat.agents.random import RandomAgent
from skat.game import GamePhase, Round
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND
from skat.models import SuitSoloNet

# block offsets, in the order Round.get_state concatenates them
HAND = slice(0, 32)
GAME = slice(32, 38)
ROLE = slice(38, 39)
SOLOIST_SEAT = slice(39, 42)
POINTS = slice(42, 44)
PLAYED = slice(44, 140)
CURRENT = slice(140, 236)
FRONT_HAND = slice(236, 239)
TRICK_VALUE = slice(239, 240)


def played_slot(state, slot):
    return state[PLAYED][32 * slot : 32 * (slot + 1)]


def current_slot(state, slot):
    return state[CURRENT][32 * slot : 32 * (slot + 1)]


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


class SeatInvarianceTest(unittest.TestCase):
    """
    The headline property: relabel the seats and the observation must not move.
    """

    def test_the_same_situation_looks_identical_from_every_chair(self) -> None:
        a, b, c, skat = DEAL
        # the same three hands, rotated onto different seats, with the soloist
        # rotated to match - strategically one and the same position
        # the dealer rotates with them: it fixes front/middle/back hand, and
        # leaving it put would place the three observers in genuinely different
        # trick positions rather than relabelled ones
        base = make_round([a, b, c, skat], soloist=0, dealer=0)
        by_one = make_round([c, a, b, skat], soloist=1, dealer=1)
        by_two = make_round([b, c, a, skat], soloist=2, dealer=2)

        first = base.get_state(0)
        np.testing.assert_array_equal(first, by_one.get_state(1))
        np.testing.assert_array_equal(first, by_two.get_state(2))

    def test_it_still_distinguishes_genuinely_different_seats(self) -> None:
        # sanity: the invariance above must not come from the vector being blind
        round_ = make_round(DEAL, soloist=0)
        self.assertFalse(
            np.array_equal(round_.get_state(0), round_.get_state(1)),
            "different hands and roles must not collapse to the same vector",
        )


class EgocentricBlocksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.round = make_round(DEAL, soloist=0, steps=6)  # two full tricks

    def test_own_played_cards_are_always_in_slot_zero(self) -> None:
        for seat in range(3):
            np.testing.assert_array_equal(
                played_slot(self.round.get_state(seat), 0),
                self.round.trick_history.to_numpy(player_id=seat),
            )

    def test_the_other_seats_follow_in_play_order(self) -> None:
        for seat in range(3):
            state = self.round.get_state(seat)
            for offset in (1, 2):
                np.testing.assert_array_equal(
                    played_slot(state, offset),
                    self.round.trick_history.to_numpy(player_id=(seat + offset) % 3),
                )

    def test_the_live_trick_is_attributed(self) -> None:
        round_ = make_round(DEAL, soloist=0, steps=7)  # one card on the table
        player, card = round_.game.trick.buffer[0]
        state = round_.get_state(player)
        self.assertEqual(1, current_slot(state, 0)[card.np_index])
        # and it is not in anyone else's slot
        self.assertEqual(0, current_slot(state, 1).sum())
        self.assertEqual(0, current_slot(state, 2).sum())


class RoleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.round = make_round(DEAL, soloist=1, steps=6)

    def test_the_soloist_knows_it_is_the_soloist(self) -> None:
        self.assertEqual(1.0, self.round.get_state(1)[ROLE][0])
        for defender in (0, 2):
            self.assertEqual(0.0, self.round.get_state(defender)[ROLE][0])

    def test_the_soloist_sits_at_a_relative_offset(self) -> None:
        for seat in range(3):
            block = self.round.get_state(seat)[SOLOIST_SEAT]
            self.assertEqual(1, block.sum())
            self.assertEqual(1.0, block[(1 - seat) % 3], f"seat {seat}")

    def test_points_are_my_party_first(self) -> None:
        solo = self.round.get_state(1)[POINTS]
        defender = self.round.get_state(0)[POINTS]
        # the same two numbers, swapped - so the pair means one thing to everyone
        np.testing.assert_allclose(solo, defender[::-1])
        np.testing.assert_allclose(
            solo, [self.round.points_soloist / 120, self.round.points_defenders / 120]
        )

    def test_an_unset_soloist_leaves_the_block_empty(self) -> None:
        # eingepasst rounds never reach play, but the encoding must not index -42
        round_ = make_round(DEAL, soloist=0)
        round_.solo_player_id = -42
        self.assertEqual(0, round_.get_state(0)[SOLOIST_SEAT].sum())


class WidthTest(unittest.TestCase):
    def test_the_vector_matches_the_network_input(self) -> None:
        """
        `Round.get_state` and `SuitSoloNet.INPUT_SIZE` are coupled by hand and
        nothing else enforces it.
        """
        self.assertEqual(SuitSoloNet.INPUT_SIZE, len(make_round(DEAL).get_state(0)))

    def test_every_game_type_yields_the_same_width(self) -> None:
        # a null game used to produce a 1-wide game block, i.e. a state vector
        # four short, which the network could not accept at all
        for game in (SuitGame(0), SuitGame(3), Grand(), Null()):
            round_ = make_round(DEAL, game=game)
            self.assertEqual(
                SuitSoloNet.INPUT_SIZE,
                len(round_.get_state(0)),
                type(game).__name__,
            )

    def test_the_game_block_identifies_the_game(self) -> None:
        for game, expected in (
            (SuitGame(0), [1, 0, 0, 0, 0, 0]),
            (SuitGame(3), [0, 0, 0, 1, 0, 0]),
            (Grand(), [0, 0, 0, 0, 1, 0]),
            (Null(), [0, 0, 0, 0, 0, 1]),
        ):
            state = make_round(DEAL, game=game).get_state(0)
            np.testing.assert_array_equal(expected, state[GAME])

    def test_everything_is_scaled_to_the_unit_interval(self) -> None:
        # one raw trick value used to sit among the one-hots, up to ~120x larger
        # than every other feature
        state = make_round(DEAL, steps=7).get_state(0)
        self.assertLessEqual(float(state.max()), 1.0)
        self.assertGreaterEqual(float(state.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
