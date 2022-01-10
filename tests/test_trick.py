import unittest

from skat.card import Card
from skat.games.suit import SuitGame
from skat.trick import Trick


class TrickTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trick = Trick(SuitGame(suit=0))

    def add_test_cards(self) -> None:
        player_cards = (
            (0, Card(3, 0)),  # value 0
            (1, Card(3, 5)),  # value 4
            (2, Card(3, 7)),  # value 11
        )
        for player, card in player_cards:
            self.trick.append(player_id=player, card=card)

    def test_empty_on_init(self) -> None:
        self.assertEqual(0, len(self.trick))

    def test_append_on_empty_trick(self) -> None:
        self.add_test_cards()
        self.assertEqual(3, len(self.trick))

    def test_append_too_many(self) -> None:
        self.add_test_cards()
        with self.assertRaises(Exception):
            self.trick.append(4, Card(0, 0))

    def test_color(self) -> None:
        self.assertIsNone(self.trick.color)
        self.add_test_cards()
        self.assertEqual('♣', self.trick.color)

    def test_value(self) -> None:
        self.add_test_cards()
        self.assertEqual(15, self.trick.value)

    def test_is_full(self) -> None:
        self.assertFalse(self.trick.is_full)
        self.add_test_cards()
        self.assertTrue(self.trick.is_full)

    def test_winner(self) -> None:
        self.add_test_cards()
        winning_player = self.trick.winner
        self.assertEqual(2, winning_player)

    def test_winner_with_empty_trick(self) -> None:
        self.assertIsNone(self.trick.winner)
