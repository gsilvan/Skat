import unittest

from skat.card import Card
from skat.games.grand import Grand


class GrandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grand = Grand()

    def test_trump_cards(self) -> None:
        expected = (Card(0, 3), Card(1, 3), Card(2, 3), Card(3, 3))
        actual = self.grand.trump_cards()
        self.assertEqual(set(expected), set(actual))

    def test_trick_winner(self) -> None:
        # Only J in trick
        cards = (Card(2, 3), Card(3, 3), Card(0, 3))
        for i, card in enumerate(cards):
            self.grand.trick.append(i, card)
        self.assertEqual(1, self.grand.trick.winner)
        # no J in trick
        self.grand.new_trick()
        cards = (Card(0, 0), Card(0, 5), Card(0, 6))
        for i, card in enumerate(cards):
            self.grand.trick.append(i, card)
        self.assertEqual(2, self.grand.trick.winner)
        # 1 J in between
        self.grand.new_trick()
        cards = (Card(2, 6), Card(0, 3), Card(2, 0))
        for i, card in enumerate(cards):
            self.grand.trick.append(i, card)
        self.assertEqual(1, self.grand.trick.winner)

    def test_forced_cards(self) -> None:
        # empty trick, no cards enforced
        self.assertEqual(set(), self.grand.trick.forced_cards)
        # add clubs card
        self.grand.trick.append(0, Card(3, 1))
        # now are clubs cards to be enforced
        clubs_set = (
            Card(3, 0),
            Card(3, 1),
            Card(3, 2),
            Card(3, 4),
            Card(3, 5),
            Card(3, 6),
            Card(3, 7),
        )
        self.assertEqual(set(clubs_set), self.grand.trick.forced_cards)
        # now test it with trumps (J)
        self.grand.new_trick()
        self.grand.trick.append(0, Card(3, 3))
        trump_set = (Card(0, 3), Card(1, 3), Card(2, 3), Card(3, 3))
        self.assertEqual(set(trump_set), self.grand.trick.forced_cards)

    def test_trick_is_trump(self) -> None:
        # empty trick is always False
        self.assertFalse(self.grand.trick.is_trump)
        # add non-trump-card
        self.grand.trick.append(0, Card(2, 4))
        self.assertFalse(self.grand.trick.is_trump)
        # do it again with trump card
        self.grand.new_trick()
        self.grand.trick.append(0, Card(1, 3))
        self.assertTrue(self.grand.trick.is_trump)
