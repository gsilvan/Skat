"""
`pysgf` is optional: only the parsing of raw `.sgf` text needs it.

These tests pin the split - what has to keep working without the dependency,
and what has to fail with an actionable message instead of an AttributeError
deep inside the parser.
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import iss
from iss import HAS_SGF_SUPPORT, GameType, ISSGame, ISSGames, MissingSgfSupport

SGF_LINE = (
    "(;GM[Skat]PC[International Skat Server]CO[]SE[344037]ID[6997010]"
    "DT[2021-04-30/01:07:29/UTC]P0[theCount]P1[blkkjk]P2[zoot]R0[]R1[0.0]R2[]"
    "MV[w HQ.HA.H7.CT.ST.SK.SA.HJ.CJ.CK.C8.DQ.S9.SQ.D9.C7.HK.DT.HT.CA.CQ.D7.DK."
    "H9.SJ.DJ.H8.S7.D8.S8.DA.C9 1 p 2 18 0 y 2 20 0 y 2 22 0 y 2 23 0 y 2 24 0 "
    "y 2 27 0 y 2 30 0 y 2 33 0 y 2 35 0 y 2 36 0 y 2 40 0 y 2 44 0 y 2 45 0 y "
    "2 46 0 y 2 p 0 s w DA.C9 0 G.H7.HQ 0 CJ 1 S9 2 DJ 0 ST 1 SQ 2 S7 0 SA 1 D9 "
    "2 S8 0 HA 1 HK 2 H8 0 CK 1 C7 2 CQ 0 C9 1 C8 2 H9 0 HJ 1 DQ 2 SJ 2 D7 0 DA "
    "1 DT 0 SK 1 HT 2 D8 0 CT 1 CA 2 DK ]"
    "R[d:0 win v:48 m:1 bidok p:88 t:8 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)"
)


def without_pysgf():
    """Pretend the optional dependency is not installed."""
    return patch.object(iss, "SGF", None)


class WorksWithoutPysgfTest(unittest.TestCase):
    """Everything that is plain Python has to stay available."""

    def test_the_package_imports(self) -> None:
        self.assertIsInstance(HAS_SGF_SUPPORT, bool)

    def test_the_capability_flag_reflects_the_import(self) -> None:
        self.assertEqual(iss.SGF is not None, HAS_SGF_SUPPORT)

    def test_game_types_are_usable(self) -> None:
        with without_pysgf():
            self.assertEqual(6, len(list(GameType)))
            self.assertIsNotNone(GameType.GRAND)

    def test_the_engine_bridge_needs_no_pysgf(self) -> None:
        # skat.utils.game_converter maps GameType -> Game and is what
        # Tournament uses to replay ISS deals
        with without_pysgf():
            from skat.games.grand import Grand
            from skat.utils.game_converter import get_game

            self.assertIsInstance(get_game(GameType.GRAND), Grand)

    def test_pkl_corpora_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.pkl"
            path.write_bytes(pickle.dumps([]))
            with without_pysgf():
                self.assertEqual(0, len(ISSGames(str(path))))

    @unittest.skipUnless(HAS_SGF_SUPPORT, "needs pysgf to build the corpus")
    def test_a_corpus_parsed_earlier_replays_without_pysgf(self) -> None:
        """The real workflow: parse once with pysgf, train forever without it."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.pkl"
            path.write_bytes(pickle.dumps([ISSGame(SGF_LINE), ISSGame(SGF_LINE)]))

            with without_pysgf():
                games = ISSGames(str(path))
                self.assertEqual(2, len(games))
                self.assertEqual(GameType.GRAND, games[0].type)
                self.assertEqual(88, games[0].points)
                self.assertEqual(32, len(games[0].deck))
                self.assertEqual(1, len(games.sample(n=1, seed=0)))


class FailsClearlyWithoutPysgfTest(unittest.TestCase):
    """The `.sgf` paths must refuse loudly, not crash obscurely."""

    def test_parsing_a_record_raises(self) -> None:
        with without_pysgf(), self.assertRaises(MissingSgfSupport):
            ISSGame(SGF_LINE)

    def test_reading_an_sgf_file_raises_before_touching_it(self) -> None:
        # The archives run to gigabytes, so the guard has to come before the
        # open(). Pointing at a missing file proves it does: were the check any
        # later, this would surface as FileNotFoundError instead.
        with without_pysgf(), self.assertRaises(MissingSgfSupport):
            ISSGames("does-not-exist.sgf")

    def test_the_message_says_how_to_fix_it(self) -> None:
        with without_pysgf(), self.assertRaises(MissingSgfSupport) as caught:
            ISSGame(SGF_LINE)
        message = str(caught.exception)
        self.assertIn("pysgf", message)
        self.assertIn("--extra sgf", message)
        self.assertIn(".pkl", message)  # points at the way out

    def test_it_is_an_import_error(self) -> None:
        # callers already guarding optional imports catch it for free
        self.assertTrue(issubclass(MissingSgfSupport, ImportError))

    def test_unknown_extensions_still_report_themselves(self) -> None:
        with without_pysgf(), self.assertRaises(Exception) as caught:
            ISSGames("corpus.txt")
        self.assertNotIsInstance(caught.exception, MissingSgfSupport)


if __name__ == "__main__":
    unittest.main()
