import unittest

from skat.utils.skatstube import SkatstubeGame

TESTFILE_FOLDER = "./testfiles/skatstube/"


class SkatstubeGameTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_parse(self) -> None:
        file_name = "340939961.json"
        stube_game = SkatstubeGame(TESTFILE_FOLDER + file_name)
        self.assertEqual("Herz", stube_game.game)

    def test_file_not_found(self) -> None:
        fail_name = "fail.json"
        with self.assertRaises(FileNotFoundError):
            SkatstubeGame(TESTFILE_FOLDER + fail_name)
