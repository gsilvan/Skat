import unittest

from skat.card import Card


class CardTest(unittest.TestCase):

    def setUp(self) -> None:
        self.lower_card = Card(0, 0)
        self.higher_card = Card(0, 1)
        self.card_1 = Card(1, 3)
        self.card_2 = Card(1, 3)
        self.card_3 = Card(2, 4)

    def test_lt_gt(self) -> None:
        self.assertTrue(self.lower_card < self.higher_card)
        self.assertFalse(self.lower_card > self.higher_card)

    def test_lt_sorting(self) -> None:
        _card_1 = Card(0, 7)  # Diamonds Ace (♦A)
        _card_2 = Card(1, 4)  # Queen of Hearts (♥Q)
        self.assertTrue(_card_1 < _card_2)

    def test_eq(self) -> None:
        self.assertTrue(self.card_1 == self.card_2)
        self.assertFalse(self.card_1 == self.card_3)
        self.assertTrue(self.card_1 != self.card_3)

    def test_str_repr(self) -> None:
        self.assertTrue(isinstance(str(self.card_1), str))
        self.assertTrue(isinstance(repr(self.card_1), str))

    def test_hash(self) -> None:
        self.assertEqual(hash(self.card_1), hash(self.card_2))
