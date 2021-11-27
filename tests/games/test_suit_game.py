import unittest

from skat.card import Card
from skat.games.suit import SuitGame


class SuitGameTest(unittest.TestCase):

    def test_find_all_trumps(self) -> None:
        for i in range(4):
            suit_game = SuitGame(i)
            trumps_test_set = [
                Card(0, 3),  # ♦J
                Card(1, 3),  # ♥J
                Card(2, 3),  # ♠J
                Card(3, 3),  # ♣J
                Card(i, 0),  # ♦7
                Card(i, 1),  # ♦8
                Card(i, 2),  # ♦9
                Card(i, 4),  # ♦Q
                Card(i, 5),  # ♦K
                Card(i, 6),  # ♦10
                Card(i, 7),  # ♦A
            ]
            trumps_test_set.sort()
            trumps = suit_game.trumps()
            trumps.sort()
            self.assertEqual(trumps_test_set, trumps)

    def test_game_values(self) -> None:
        test_set = (9, 10, 11, 12)
        for i in range(4):
            suit_game = SuitGame(i)
            self.assertEqual(test_set[i], suit_game.value)
