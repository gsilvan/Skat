import unittest

from skat.card import Card
from skat.games.suit import SuitGame


class SuitGameTest(unittest.TestCase):

    def test_find_all_trump_cards(self) -> None:
        for i in range(4):
            suit_game = SuitGame(i)
            trumps_test_set = (
                Card(i, 0),  # ♦7
                Card(i, 1),  # ♦8
                Card(i, 2),  # ♦9
                Card(i, 4),  # ♦Q
                Card(i, 5),  # ♦K
                Card(i, 6),  # ♦10
                Card(i, 7),  # ♦A
                Card(0, 3),  # ♦J
                Card(1, 3),  # ♥J
                Card(2, 3),  # ♠J
                Card(3, 3),  # ♣J
            )
            computed_set = suit_game.trump_set()
            self.assertEqual(trumps_test_set, computed_set)

    def test_find_all_suit_cards(self) -> None:
        for i in range(4):
            suit_test_set = (
                Card(i, 0),  # ♦7
                Card(i, 1),  # ♦8
                Card(i, 2),  # ♦9
                Card(i, 4),  # ♦Q
                Card(i, 5),  # ♦K
                Card(i, 6),  # ♦10
                Card(i, 7),  # ♦A
            )
            computed_set = SuitGame.suit_set(i)
            self.assertEqual(suit_test_set, computed_set)

    def test_game_values(self) -> None:
        test_set = (9, 10, 11, 12)
        for i in range(4):
            suit_game = SuitGame(i)
            self.assertEqual(test_set[i], suit_game.value)

    def test_is_trump_card(self) -> None:
        clubs_game = SuitGame(3)
        test_trump_cards = (
            Card(3, 0),
            Card(3, 7),
            Card(0, 3),  # This is a J
            Card(1, 3),
            Card(2, 3),
            Card(3, 3),
        )
        computed_trump_cards = clubs_game.trump_set()
        # compare sets
        self.assertTrue(
            frozenset(test_trump_cards) <= frozenset(computed_trump_cards))
        # compare with function
        for trump_card in test_trump_cards:
            self.assertTrue(clubs_game.is_trump_card(trump_card))

    def test_is_not_trump_card(self) -> None:
        hearts_game = SuitGame(1)
        test_non_trump_cards = (
            Card(0, 5),
            Card(0, 1),
            Card(2, 4),
            Card(3, 1),
        )
        for non_trump_card in test_non_trump_cards:
            self.assertFalse(hearts_game.is_trump_card(non_trump_card))
