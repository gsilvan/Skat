import os
import unittest

from skat.card import Card
from skat.utils.skatstube import GameType, SkatstubeGame

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), "testfiles/skatstube/")


class SkatstubeGameTest(unittest.TestCase):
    def setUp(self) -> None:
        file_name = "340939961.json"
        self.stube_game = SkatstubeGame(os.path.join(TESTFILE_FOLDER, file_name))

    def test_parse(self) -> None:
        self.assertEqual("Herz", self.stube_game.game_type)

    def test_file_not_found(self) -> None:
        fail_name = "fail.json"
        with self.assertRaises(FileNotFoundError):
            SkatstubeGame(TESTFILE_FOLDER + fail_name)

    def test_get_hand(self) -> None:
        expected = [
            Card(1, 4),
            Card(1, 0),
            Card(1, 1),  # H8
            # Card(3, 7), EA
            Card(3, 6),
            Card(3, 5),
            Card(2, 7),
            Card(2, 6),
            Card(2, 4),
            # Card(0, 7), SA
            Card(0, 6),  # Sx
            Card(0, 5),
        ]  # no EA, SA
        hand = self.stube_game.get_hand()
        self.assertEqual(set(expected), set(hand))

    def test_get_type(self) -> None:
        expected_game_type = GameType["Herz"]
        self.assertEqual(expected_game_type, self.stube_game.get_type())

    def test_hand_reconstruction(self) -> None:
        expected_hand = (
            ("H7", "H8", "HO", "EX", "EK", "GA", "GX", "GO", "SX", "SK"),
            ("EU", "GU", "HU", "HK", "H9", "EO", "G9", "G8", "S8", "S7"),
            ("SU", "HA", "HX", "E9", "E8", "E7", "GK", "G7", "SO", "S9"),
        )
        for i, h in enumerate(expected_hand):
            self.assertTrue(set(h) == set(self.stube_game.hand[i]))
