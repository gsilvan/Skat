import unittest

from skat.card import Card
from skat.games.null import Null, NullTrick


class NullTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Null()

    def test_trump_cards(self) -> None:
        self.assertEqual(set(), set(self.game.trump_cards()))

    def test_trick_winner(self) -> None:
        # s9, s10, sJ
        cards = (Card(2, 2), Card(2, 6), Card(2, 3))
        for i, card in enumerate(cards):
            self.game.trick.append(i, card)
        self.assertEqual(2, self.game.trick.winner)
        # c8, c7, hA
        self.game.new_trick()
        cards = (Card(3, 1), Card(3, 0), Card(1, 7))
        for i, card in enumerate(cards):
            self.game.trick.append(i, card)
        self.assertEqual(0, self.game.trick.winner)
        # trick is not full
        self.game.new_trick()
        self.game.trick.append(0, Card(2, 2))
        self.assertIsNone(self.game.trick.winner)

    def test_forced_cards(self) -> None:
        # empty trick, no cards enforced
        self.assertEqual(set(), self.game.trick.forced_cards)
        # add hearts card
        self.game.trick.append(0, Card(1, 5))
        hearts_cards = [Card(1, j) for j in range(8)]
        self.assertEqual(set(hearts_cards), set(self.game.trick.forced_cards))

    def test_better_than(self) -> None:
        self.assertTrue(NullTrick.better_than(Card(0, 1), Card(0, 0)))
        self.assertTrue(NullTrick.better_than(Card(0, 3), Card(0, 0)))
        self.assertTrue(NullTrick.better_than(Card(0, 3), Card(0, 6)))

    def test_trick_is_trump(self) -> None:
        self.game.trick.append(0, Card(3, 3))
        self.assertFalse(self.game.trick.is_trump)  # null has no trump
