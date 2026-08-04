import unittest

from skat.agents.search.playout import PlayoutState, copy_trick
from skat.card import Card
from skat.games.null import NullTrick
from skat.games.suit import SuitGameTrick

CLUBS = 3


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def state(hands, to_move=0, points=None, trick=None, **kwargs) -> PlayoutState:
    return PlayoutState(
        trick=trick if trick is not None else SuitGameTrick(CLUBS),
        hands=hands,
        to_move=to_move,
        points=points if points is not None else [0, 0, 0],
        soloist=0,
        **kwargs,
    )


class CopyTrickTest(unittest.TestCase):
    def test_keeps_rule_configuration_and_detaches_buffer(self) -> None:
        original = SuitGameTrick(CLUBS)
        original.append(0, card("♦", "A"))
        clone = copy_trick(original)
        assert isinstance(clone, SuitGameTrick)
        self.assertEqual(CLUBS, clone.trump_suit)
        clone.append(1, card("♦", "K"))
        self.assertEqual(1, len(original))

    def test_can_drop_the_buffer(self) -> None:
        original = SuitGameTrick(CLUBS)
        original.append(0, card("♦", "A"))
        self.assertEqual(0, len(copy_trick(original, keep_buffer=False)))


class LegalMovesTest(unittest.TestCase):
    def test_leading_is_unrestricted(self) -> None:
        hand = [card("♦", "A"), card("♥", "7")]
        self.assertEqual(set(hand), set(state([hand, [], []]).legal_moves()))

    def test_must_follow_suit(self) -> None:
        trick = SuitGameTrick(CLUBS)
        trick.append(2, card("♦", "A"))
        hand = [card("♦", "7"), card("♥", "7"), card("♣", "7")]
        moves = state([hand, [], []], trick=trick).legal_moves()
        self.assertEqual([card("♦", "7")], moves)

    def test_free_choice_when_void(self) -> None:
        trick = SuitGameTrick(CLUBS)
        trick.append(2, card("♦", "A"))
        hand = [card("♥", "7"), card("♣", "7")]
        moves = state([hand, [], []], trick=trick).legal_moves()
        self.assertEqual(set(hand), set(moves))

    def test_jacks_are_trump_not_their_suit(self) -> None:
        trick = SuitGameTrick(CLUBS)
        trick.append(2, card("♦", "A"))
        # ♦J is a club (trump) here, so it does not follow diamonds.
        hand = [card("♦", "J"), card("♥", "7")]
        moves = state([hand, [], []], trick=trick).legal_moves()
        self.assertEqual(set(hand), set(moves))


class ApplyTest(unittest.TestCase):
    def test_incomplete_trick_advances_seat(self) -> None:
        s = state([[card("♦", "A")], [], []], to_move=0)
        s.apply(card("♦", "A"))
        self.assertEqual(1, s.to_move)
        self.assertEqual(1, len(s.trick))

    def test_full_trick_books_points_and_leads_the_winner(self) -> None:
        s = state(
            [[card("♦", "7")], [card("♦", "A")], [card("♦", "X")]],
            to_move=0,
        )
        for seat in (0, 1, 2):
            s.apply(s.hands[seat][0])
        self.assertEqual(1, s.to_move)  # ♦A took it
        self.assertEqual([0, 21, 0], s.points)
        self.assertEqual(0, len(s.trick))
        self.assertTrue(s.is_terminal)

    def test_trump_beats_the_led_suit(self) -> None:
        s = state(
            [[card("♦", "7")], [card("♦", "A")], [card("♣", "7")]],
            to_move=0,
        )
        for seat in (0, 1, 2):
            s.apply(s.hands[seat][0])
        self.assertEqual(2, s.to_move)
        self.assertEqual([0, 0, 11], s.points)

    def test_clone_is_independent(self) -> None:
        s = state([[card("♦", "7"), card("♥", "7")], [], []])
        clone = s.clone()
        clone.apply(card("♦", "7"))
        self.assertEqual(2, len(s.hands[0]))
        self.assertEqual(0, len(s.trick))


class RewardTest(unittest.TestCase):
    def test_soloist_needs_61(self) -> None:
        won = state([[], [], []], points=[61, 59, 0]).rewards()
        lost = state([[], [], []], points=[60, 60, 0]).rewards()
        self.assertGreater(won[0], lost[0])
        self.assertGreater(won[0], 0.9)
        self.assertLess(lost[0], 0.1)

    def test_defenders_get_the_complement(self) -> None:
        rewards = state([[], [], []], points=[75, 45, 0]).rewards()
        self.assertAlmostEqual(1.0, rewards[0] + rewards[1])
        self.assertEqual(rewards[1], rewards[2])

    def test_more_points_is_better_within_a_win(self) -> None:
        self.assertGreater(
            state([[], [], []], points=[100, 20, 0]).rewards()[0],
            state([[], [], []], points=[70, 50, 0]).rewards()[0],
        )

    def test_null_soloist_wins_by_taking_nothing(self) -> None:
        clean = state([[], [], []], trick=NullTrick(), is_null=True).rewards()
        self.assertEqual(1.0, clean[0])
        self.assertEqual(0.0, clean[1])

    def test_null_soloist_loses_on_any_trick(self) -> None:
        # points are irrelevant in null - a single trick loses it
        taken = state(
            [[], [], []],
            points=[2, 118, 0],
            trick=NullTrick(),
            is_null=True,
            soloist_took_trick=True,
        ).rewards()
        self.assertEqual(0.0, taken[0])
        self.assertEqual(1.0, taken[1])


if __name__ == "__main__":
    unittest.main()
