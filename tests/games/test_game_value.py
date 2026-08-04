import unittest

from skat.card import Card
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def hand_of(*names: str) -> set[Card]:
    return {card(name[0], name[1:]) for name in names}


class MatadorTest(unittest.TestCase):
    def test_mit_counts_the_run_we_hold(self) -> None:
        # ♦J missing breaks the run after three
        self.assertEqual(3, SuitGame(0).matadors(hand_of("♣J", "♠J", "♥J", "♦7")))

    def test_ohne_counts_the_run_we_lack(self) -> None:
        # no ♣J -> ohne 1, the run of missing top trumps stops at ♠J
        self.assertEqual(1, SuitGame(0).matadors(hand_of("♠J", "♦7")))

    def test_the_run_continues_into_the_suit_cards(self) -> None:
        # no jack and no ♦A/♦X, but we hold ♦K -> ohne 6
        self.assertEqual(6, SuitGame(0).matadors(hand_of("♦7", "♦8", "♦9", "♦Q", "♦K")))

    def test_grand_only_ranks_the_jacks(self) -> None:
        self.assertEqual(2, Grand().matadors(hand_of("♣J", "♠J", "♦A")))

    def test_holding_every_trump_is_mit_eleven(self) -> None:
        every = hand_of(
            "♣J", "♠J", "♥J", "♦J", "♦A", "♦X", "♦K", "♦Q", "♦9", "♦8", "♦7"
        )
        self.assertEqual(11, SuitGame(0).matadors(every))

    def test_null_has_no_matadors(self) -> None:
        self.assertEqual(0, Null().matadors(hand_of("♣J", "♠J")))


class SpielwertTest(unittest.TestCase):
    def test_suit_game_scales_with_the_matadors(self) -> None:
        # ohne 6 in diamonds -> 7 x 9
        self.assertEqual(
            63, SuitGame(0).spielwert(hand_of("♦7", "♦8", "♦9", "♦Q", "♦K"))
        )

    def test_the_multiplier_follows_the_suit(self) -> None:
        cards = hand_of("♠J", "♦7")  # ohne 1 -> 2 x base
        self.assertEqual(18, SuitGame(0).spielwert(cards))  # ♦ 9
        self.assertEqual(20, SuitGame(1).spielwert(cards))  # ♥ 10
        self.assertEqual(22, SuitGame(2).spielwert(cards))  # ♠ 11
        self.assertEqual(24, SuitGame(3).spielwert(cards))  # ♣ 12

    def test_grand_uses_its_own_base(self) -> None:
        # mit 3 -> 4 x 24
        self.assertEqual(96, Grand().spielwert(hand_of("♣J", "♠J", "♥J", "♦A")))

    def test_null_values_are_flat(self) -> None:
        cards = hand_of("♣J", "♠J", "♥J")
        self.assertEqual(23, Null().spielwert(cards))
        self.assertEqual(35, Null(hand=True).spielwert(cards))
        self.assertEqual(46, Null(ouvert=True).spielwert(cards))
        self.assertEqual(59, Null(hand=True, ouvert=True).spielwert(cards))

    def test_the_skat_can_lower_the_value(self) -> None:
        # Ohne 6 (the run ♣J ♠J ♥J ♦J ♣A ♣X is missing) is worth 7 x 12 = 84.
        # Finding ♥J in the skat cuts the run to ohne 2, i.e. 3 x 12 = 36 -
        # which is exactly how a soloist who bid 40 ends up overbid.
        hand = hand_of("♣7", "♣8", "♣9", "♣Q", "♣K")
        self.assertEqual(84, SuitGame(3).spielwert(hand))
        self.assertEqual(36, SuitGame(3).spielwert(hand | hand_of("♥J")))


if __name__ == "__main__":
    unittest.main()
