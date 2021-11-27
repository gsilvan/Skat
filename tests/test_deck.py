import unittest

from skat.deck import Deck


class DeckTest(unittest.TestCase):

    def setUp(self) -> None:
        self.deck = Deck()

    def test_str_repr(self) -> None:
        self.assertTrue(isinstance(str(self.deck), str))

    def test_empty_deck(self) -> None:
        self.assertEqual(0, len(self.deck))

    def test_initialized_deck(self) -> None:
        self.deck.initialize_cards()
        self.assertEqual(32, len(self.deck))

    def test_hash(self) -> None:
        self.deck.initialize_cards()
        _deck = Deck()
        _deck.initialize_cards()
        self.assertEqual(hash(self.deck), hash(_deck))
        _deck.shuffle()
        self.assertNotEqual(hash(self.deck), hash(_deck))

    def test_basic_dealing(self) -> None:
        self.deck.initialize_cards()
        self.assertEqual(32, len(self.deck))
        hand = self.deck.deal_cards()
        self.assertEqual(10, len(hand))
        self.assertEqual(22, len(self.deck))
