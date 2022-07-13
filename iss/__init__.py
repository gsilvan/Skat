#!/usr/bin/env python3
import enum
import re
from typing import Optional

from pysgf import SGF

from skat.deck import Deck
from skat.card import Card

card = {
    "D7": Card(0, 0),
    "D8": Card(0, 1),
    "D9": Card(0, 2),
    "DJ": Card(0, 3),
    "DQ": Card(0, 4),
    "DK": Card(0, 5),
    "DT": Card(0, 6),
    "DA": Card(0, 7),
    "H7": Card(1, 0),
    "H8": Card(1, 1),
    "H9": Card(1, 2),
    "HJ": Card(1, 3),
    "HQ": Card(1, 4),
    "HK": Card(1, 5),
    "HT": Card(1, 6),
    "HA": Card(1, 7),
    "S7": Card(2, 0),
    "S8": Card(2, 1),
    "S9": Card(2, 2),
    "SJ": Card(2, 3),
    "SQ": Card(2, 4),
    "SK": Card(2, 5),
    "ST": Card(2, 6),
    "SA": Card(2, 7),
    "C7": Card(3, 0),
    "C8": Card(3, 1),
    "C9": Card(3, 2),
    "CJ": Card(3, 3),
    "CQ": Card(3, 4),
    "CK": Card(3, 5),
    "CT": Card(3, 6),
    "CA": Card(3, 7),
}


class GameType(enum.Enum):
    DIAMONDS = enum.auto()
    HEARTS = enum.auto()
    SPADES = enum.auto()
    CLUBS = enum.auto()
    GRAND = enum.auto()
    NULL = enum.auto()


class ISSGame:
    """
    A ISSGame represents a single game played on the International Skat Server (ISS)
    https://skatgame.net/iss/
    """

    def __init__(self, game_str: str) -> None:
        self._deck = None
        self._is_valid = None
        self._is_won = None
        self._points = None
        self._soloist = None
        self._type = None
        self._value = None
        self._parse_sgf_line(game_str)

    @property
    def deck(self) -> Optional[Deck]:
        return self._deck

    @property
    def hand_game(self) -> Optional[bool]:
        """Declarer picked up the skat. Note: We consider all games as Hand games."""
        return True

    @property
    def points(self) -> Optional[int]:
        return self._points

    @property
    def soloist(self) -> Optional[int]:
        return self._soloist

    @property
    def is_valid(self) -> Optional[bool]:
        return self._is_valid

    @property
    def type(self) -> Optional[GameType]:
        return

    @property
    def value(self) -> Optional[int]:
        return self._value

    @property
    def is_won(self):
        return self._is_won

    def _parse_sgf_line(self, sgf_line: str):
        root = SGF.parse(sgf_line)
        __mv = str(root.get_list_property("MV"))
        deck_s = __mv.split(" ")[1].split(".")
        declare_s = re.search(r"([DHSCGN].[DHSC][789TJQKA].[DHSC][789TJQKA])", __mv)[
            0
        ].split(".")
        match declare_s[0]:
            case "D":
                self._type = GameType.DIAMONDS
            case "H":
                self._type = GameType.HEARTS
            case "S":
                self._type = GameType.SPADES
            case "C":
                self._type = GameType.CLUBS
            case "G":
                self._type = GameType.GRAND
            case "N":
                self._type = GameType.NULL

        __r = str(root.get_list_property("R"))
        if re.search("win", __r):
            self._is_won = True
            self._is_valid = True
        if re.search("loss", __r):
            self._is_won = False
            self._is_valid = True
        if re.search("penalty", __r):
            self._is_won = False
            self._is_valid = False

        self._soloist = int(re.search(r"(d:-?[012])", __r)[0].split(":")[1])
        self._points = int(re.search(r"(p:[0-9]{1,3})", __r)[0].split(":")[1])
        self._value = int(re.search(r"(v:[0-9]{1,3})", __r)[0].split(":")[1])

        hands = [
            [card[c] for c in deck_s[0:10]],
            [card[c] for c in deck_s[10:20]],
            [card[c] for c in deck_s[20:30]],
        ]
        skat_received = [card[c] for c in deck_s[30:32]]
        hands[self._soloist].extend(skat_received)
        skat_put = [card[c] for c in declare_s[1:]]
        for c in skat_put:
            hands[self._soloist].remove(c)
        hands.append(skat_put)
        self._deck = Deck.factory(*hands)


if __name__ == "__main__":
    iss_game = ISSGame(
        "(;GM[Skat]PC[International Skat Server]CO[]SE[344037]ID[6997010]DT[2021-04-30/01:07:29/UTC]P0[theCount]P1[blkkjk]P2[zoot]R0[]R1[0.0]R2[]MV[w HQ.HA.H7.CT.ST.SK.SA.HJ.CJ.CK.C8.DQ.S9.SQ.D9.C7.HK.DT.HT.CA.CQ.D7.DK.H9.SJ.DJ.H8.S7.D8.S8.DA.C9 1 p 2 18 0 y 2 20 0 y 2 22 0 y 2 23 0 y 2 24 0 y 2 27 0 y 2 30 0 y 2 33 0 y 2 35 0 y 2 36 0 y 2 40 0 y 2 44 0 y 2 45 0 y 2 46 0 y 2 p 0 s w DA.C9 0 G.H7.HQ 0 CJ 1 S9 2 DJ 0 ST 1 SQ 2 S7 0 SA 1 D9 2 S8 0 HA 1 HK 2 H8 0 CK 1 C7 2 CQ 0 C9 1 C8 2 H9 0 HJ 1 DQ 2 SJ 2 D7 0 DA 1 DT 0 SK 1 HT 2 D8 0 CT 1 CA 2 DK ]R[d:0 win v:48 m:1 bidok p:88 t:8 s:0 z:0 p0:0 p1:0 p2:0 l:-1 to:-1 r:0] ;)"
    )
    print(iss_game.deck)
    print(iss_game.soloist)
    print(iss_game.is_won)
