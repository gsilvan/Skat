import unittest

from iss import GameType, ISSGame
from skat.card import Card


class ISSTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_easy_example(self) -> None:
        iss_game = ISSGame(
            "(;GM[Skat]PC[International Skat Server]CO[]SE[344037]ID[6997010]DT[2021-04-30/01:07:29/UTC]P0[theCount]P1[blkkjk]P2[zoot]R0[]R1[0.0]R2[]MV[w HQ.HA.H7.CT.ST.SK.SA.HJ.CJ.CK.C8.DQ.S9.SQ.D9.C7.HK.DT.HT.CA.CQ.D7.DK.H9.SJ.DJ.H8.S7.D8.S8.DA.C9 1 p 2 18 0 y 2 20 0 y 2 22 0 y 2 23 0 y 2 24 0 y 2 27 0 y 2 30 0 y 2 33 0 y 2 35 0 y 2 36 0 y 2 40 0 y 2 44 0 y 2 45 0 y 2 46 0 y 2 p 0 s w DA.C9 0 G.H7.HQ 0 CJ 1 S9 2 DJ 0 ST 1 SQ 2 S7 0 SA 1 D9 2 S8 0 HA 1 HK 2 H8 0 CK 1 C7 2 CQ 0 C9 1 C8 2 H9 0 HJ 1 DQ 2 SJ 2 D7 0 DA 1 DT 0 SK 1 HT 2 D8 0 CT 1 CA 2 DK ]R[d:0 win v:48 m:1 bidok p:88 t:8 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)"
        )
        exp_h0 = {
            Card(1, 7),
            Card(3, 6),
            Card(2, 6),
            Card(2, 5),
            Card(2, 7),
            Card(1, 3),
            Card(3, 3),
            Card(3, 5),
            Card(0, 7),
            Card(3, 2),
        }
        self.assertEqual(exp_h0, set(iss_game.deck[:10]))
        # TODO: ^this line correlates with test error in deck
        self.assertEqual(True, iss_game.is_valid)
        self.assertEqual(True, iss_game.is_won)
        self.assertEqual(88, iss_game.points)
        self.assertEqual(0, iss_game.soloist)
        self.assertEqual(GameType.GRAND, iss_game._type)
        self.assertEqual(48, iss_game.value)

    def random_examples(self) -> None:
        testdata = (
            (
                "(;GM[Skat]PC[International Skat Server]CO[]SE[341721]ID[6934288]DT[2021-03-27/11:15:39/UTC]P0[xskat]P1[bernie]P2[kafi]R0[]R1[]R2[0.0]MV[w SK.CA.S7.DT.ST.S9.C7.HJ.H7.C8.HA.SA.D9.SQ.CK.S8.HK.CQ.H9.HT.DJ.D8.SJ.C9.DK.CT.DA.H8.CJ.HQ.DQ.D7 1 p 2 18 0 p 2 s w DQ.D7 2 G.C9.CT 0 CA 1 CK 2 DJ 2 SJ 0 HJ 1 S8 2 DA 0 DT 1 D9 2 H8 0 H7 1 HT 1 HA 2 HQ 0 ST 1 SA 2 CJ 0 S7 2 DQ 0 C7 1 H9 2 SC 0 RE 1 RE ]R[d:2 win v:72 m:2 bidok p:86 t:8 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:1] ;)",
                True,
                True,
                86,
                2,
                GameType.GRAND,
                72,
            ),
            (
                "(;GM[Skat]PC[International Skat Server]CO[]SE[314489]ID[6326040]DT[2020-04-28/16:34:10/UTC]P0[xskat]P1[Allen14]P2[bernie]R0[]R1[0.0]R2[]MV[w C9.SQ.CA.S8.SJ.DT.CJ.SK.DK.D9.DJ.S7.H8.D8.HQ.HJ.S9.D7.C8.HK.ST.DQ.CT.HA.HT.CQ.H7.CK.DA.H9.C7.SA 1 p 2 p 0 18 0 s w C7.SA 0 S.C9.C7 0 SJ 1 S7 2 ST 0 CJ 1 S9 2 H7 0 S8 1 DJ 2 CT 1 HQ 2 HA 0 SA 0 SQ 1 HJ 2 HT 1 HK 2 H9 0 SK 0 CA 1 C8 2 CQ 0 D9 1 D8 2 DQ 2 DA 0 DK 1 D7 2 CK 0 DT 1 H8 ]R[d:0 win v:33 m:2 bidok p:61 t:5 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)",
                True,
                True,
                33,
                0,
                GameType.SPADES,
                61,
            ),
        )
        for t in testdata:
            iss_game = ISSGame(t[0])
            self.assertEqual(t[1], iss_game.is_valid)
            self.assertEqual(t[2], iss_game.is_won)
            self.assertEqual(t[3], iss_game.points)
            self.assertEqual(t[4], iss_game.soloist)
            self.assertEqual(t[5], iss_game._type)
            self.assertEqual(t[6], iss_game.value)

    def test_buggy_1(self) -> None:
        # just has to work
        game_str = "(;GM[Skat]PC[Internet Skat Server]CO[]SE[126]ID[25]DT[2007-11-02/15:24:24/UTC]P0[Patrickm]P1[foo]P2[bonsai]R0[0.0]R1[0.0]R2[0.0]MV[w CJ.S7.H8.SA.DA.CQ.DK.CK.C8.D8.D9.HK.HA.SQ.CT.S8.HJ.SJ.DT.ST.H9.S9.HQ.SK.HT.CA.C9.C7.D7.DQ.H7.DJ 1 18 0 p 2 p 1 NOH 0 C8 1 CT 2 C9 ]R[d:1 loss v:-118 m:0 bidok p:12 t:1 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)"
        ISSGame(game_str)

    def test_buggy_2(self) -> None:
        # just has to work
        game_str = "(;GM[Skat]PC[Internet Skat Server]CO[]SE[1901]ID[13577]DT[2008-05-03/14:17:03/UTC]P0[bert]P1[Madmax]P2[ernie]R0[0.0]R1[0.0]R2[0.0]MV[w H9.HT.C8.SA.D8.DQ.CQ.SJ.SQ.CK.S8.CT.C7.CA.C9.D7.CJ.H7.H8.HJ.HA.HK.S9.DT.S7.DJ.DK.SK.D9.DA.HQ.ST 1 18 0 p 2 p 1 CO 0 SA 1 S8 2 SK ]R[d:1 loss v:-192 m:1 bidok p:13 t:0 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)"
        ISSGame(game_str)
