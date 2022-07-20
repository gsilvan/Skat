#!/usr/bin/env python3
import collections.abc
import enum
import os
import pickle
import re
from typing import Optional
from random import Random

import tqdm
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
        self._hand_game = None
        self._is_valid = None
        self._is_won = None
        self._passed = None
        self._points = None
        self._soloist = None
        self._type = None
        self._value = None
        self.__game_str = game_str
        self._parse_sgf_line(self.__game_str)

    def __str__(self) -> str:
        return self.__game_str

    @property
    def deck(self) -> Optional[Deck]:
        return self._deck

    @property
    def hand_game(self) -> Optional[bool]:
        """Declarer picked up the skat."""
        return self._hand_game

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
        return self._type

    @property
    def value(self) -> Optional[int]:
        return self._value

    @property
    def is_won(self):
        return self._is_won

    def _parse_sgf_line(self, sgf_line: str):
        root = SGF.parse(sgf_line)

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
            return  # we exit here because game is invalid
        if re.search("passed", __r):
            self._is_valid = False  # TODO: i'm lazy
            self._passed = True
            return  # we exit here because game was passed

        __mv = str(root.get_list_property("MV")).replace("|", ".")  # introduced in 2011
        if re.search("(TI.2)|(LE.2)|(LE.1)", __mv):
            self._is_valid = False
            return  # something is worng, get the hell out
        deck_s = __mv.split(" ")[1].split(".")
        dec_search_result = re.search(
            r"([DHSCGN]O?.[DHSC][789TJQKA].[DHSC][789TJQKA])|([DHSCGN]O?H)|([DHSCGN]O)",
            __mv,
        )
        declare_s = None
        if dec_search_result:
            declare_s = dec_search_result[0].split(".")
        elif not dec_search_result:
            self._is_valid = False
            return  # something is worng, get the hell out
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
            case "DH" | "DO" | "DOH":
                self._type = GameType.DIAMONDS
                self._hand_game = True
            case "HH" | "HO" | "HOH":
                self._type = GameType.HEARTS
                self._hand_game = True
            case "SH" | "SO" | "SOH":
                self._type = GameType.SPADES
                self._hand_game = True
            case "CH" | "CO" | "COH":
                self._type = GameType.CLUBS
                self._hand_game = True
            case "GH" | "GO" | "GOH":
                self._type = GameType.GRAND
                self._hand_game = True
            case "NH" | "NO" | "NOH":
                self._type = GameType.NULL
                self._hand_game = True

        self._soloist = int(re.search(r"(d:-?[012])", __r)[0].split(":")[1])
        self._points = int(re.search(r"(p:[0-9]{1,3})", __r)[0].split(":")[1])
        self._value = int(re.search(r"(v:-?[0-9]{1,3})", __r)[0].split(":")[1])

        hands = [
            [card[c] for c in deck_s[0:10]],
            [card[c] for c in deck_s[10:20]],
            [card[c] for c in deck_s[20:30]],
        ]
        skat_received = [card[c] for c in deck_s[30:32]]
        if self.hand_game:
            hands.append(skat_received)
        if not self.hand_game:
            hands[self._soloist].extend(skat_received)
            skat_put = [card[c] for c in declare_s[1:]]
            for c in skat_put:
                hands[self._soloist].remove(c)
            hands.append(skat_put)
        self._deck = Deck.factory(*hands)


class ISSGames(collections.abc.MutableSequence):
    def __init__(self, file_name: str):
        _, file_extension = os.path.splitext(file_name)
        self.__game_list: list[ISSGame] = []
        match file_extension:
            case ".pkl":
                self.read_pkl(file_name)
            case ".sgf":
                self.read_sgf(file_name)
            case _:
                raise Exception("Unknown extension.")

    def __getitem__(self, i: int) -> ISSGame:
        return self.__game_list[i]

    def __setitem__(self, i: int, o: ISSGame) -> None:
        self.__game_list[i] = o

    def __delitem__(self, i: int) -> None:
        del self.__game_list[i]

    def __len__(self) -> int:
        return len(self.__game_list)

    def insert(self, index: int, value: ISSGame) -> None:
        self.__game_list.insert(index, value)

    def sample(self, n=10000, seed=None) -> list[ISSGame]:
        rand = Random(seed)
        sample = rand.sample(self.__game_list, n)
        return sample

    def read_sgf(self, filename: str) -> None:
        print("parsing iss games... ")
        with tqdm.tqdm(total=os.path.getsize(filename)) as progress:
            with open(filename, "r") as fp:
                for line in fp:
                    game = ISSGame(line.rstrip())
                    if game.is_valid:
                        self.__game_list.append(game)
                    progress.update(len(line.encode("utf-8")))

    def read_pkl(self, filename: str) -> None:
        with open(filename, "rb") as fp:
            self.__game_list = pickle.load(fp)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as fp:
            pickle.dump(self.__game_list, fp)


if __name__ == "__main__":
    # Example usage
    iss_games = ISSGames("/home/silvan/Downloads/iss-games-04-2021.sgf")
    iss_games.save("/home/silvan/Desktop/iss_games.pkl")
    my_sample = iss_games.sample(200)
