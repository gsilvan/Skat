import os
import unittest

from skat.card import Card
from skat.deck import Deck
from skat.utils.skatstube import SkatstubeGame

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), "testfiles/skatstube/")


class SkatstubeGameTest(unittest.TestCase):
    def setUp(self) -> None:
        file_name = "340939961.json"
        self.stube_game = SkatstubeGame(TESTFILE_FOLDER + file_name)

    def test_parse(self) -> None:
        self.assertEqual("Herz", self.stube_game.game)

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