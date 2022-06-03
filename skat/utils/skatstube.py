#!/usr/bin/env python3

import json
import os
import sys
from enum import Enum

from skat.card import Card
from skat.deck import Deck


class GameType(Enum):
    Karo = (0,)
    Herz = (1,)
    Pik = (2,)
    Kreuz = (3,)
    Grand = (4,)
    Null = (5,)


class SkatstubeGame:
    def __init__(self, file_name) -> None:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"{file_name} was not found.")
        with open(file_name, "r") as fp:
            self.__raw_data = json.load(fp)
        self.__parse_game()

    def __str__(self) -> str:
        return (
            f"r={self.result}\n"
            f"pts={self.points}\n"
            f"received_cards={self.skat_cards}\n"
            f"dropped_cards={self.dropped_cards}\n"
            f"final_hand={self.soloist_hand}"
        )

    def __parse_game(self) -> None:
        beginner = None
        self.hand = [[], [], []]
        for entry in self.__raw_data:
            if entry["type"] == "yourAuthenticationSucceeded":
                self.position = entry["position"]
                beginner = entry["position"]
            if entry["type"] == "youGotCards":
                self.cards = entry["cards"]
            if entry["type"] == "playsTheGame":
                self.game = entry["gameType"]
            if entry["type"] == "gameResult":
                self.result = entry["won"]
                self.points = entry["points"]
                self.skat_cards = entry["skatCards"]
                self.dropped_cards = entry["droppedCards"]
                self.soloist_hand = self.cards.copy()
                self.soloist_hand = self.soloist_hand + self.skat_cards
                for card in self.dropped_cards:
                    self.soloist_hand.remove(card)
            if entry["type"] == "wonTheTrick":
                # opponent team has to be reconstructed from the data
                trick = entry["cards"]
                for i, card in enumerate(trick):
                    self.hand[(beginner + i) % 3].append(card)
                beginner = entry["position"]  # set the beginner of the next trick

    def get_hand(self) -> list[Card]:
        hand = list()
        for card_str in self.soloist_hand:
            hand.append(CARD[card_str])
        return hand

    def get_deck(self) -> Deck:
        """Returns a Deck from Skatstube.de game data."""
        _p0_hand, _p1_hand, _p2_hand = None, None, None
        match self.position:
            case 0:
                _p0_hand = self.soloist_hand
            case 1:
                _p1_hand = self.soloist_hand
            case 2:
                _p2_hand = self.soloist_hand
        return Deck.factory(_p0_hand, _p1_hand, _p2_hand)

    def get_type(self) -> GameType:
        """Returns announced game type."""
        return GameType[self.game]


CARD = {
    "S7": Card(0, 0),
    "S8": Card(0, 1),
    "S9": Card(0, 2),
    "SU": Card(0, 3),
    "SO": Card(0, 4),
    "SK": Card(0, 5),
    "SX": Card(0, 6),
    "SA": Card(0, 7),
    "H7": Card(1, 0),
    "H8": Card(1, 1),
    "H9": Card(1, 2),
    "HU": Card(1, 3),
    "HO": Card(1, 4),
    "HK": Card(1, 5),
    "HX": Card(1, 6),
    "HA": Card(1, 7),
    "G7": Card(2, 0),
    "G8": Card(2, 1),
    "G9": Card(2, 2),
    "GU": Card(2, 3),
    "GO": Card(2, 4),
    "GK": Card(2, 5),
    "GX": Card(2, 6),
    "GA": Card(2, 7),
    "E7": Card(3, 0),
    "E8": Card(3, 1),
    "E9": Card(3, 2),
    "EU": Card(3, 3),
    "EO": Card(3, 4),
    "EK": Card(3, 5),
    "EX": Card(3, 6),
    "EA": Card(3, 7),
}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("provide 1 arg, the filename")
        exit(1)
    parser = SkatstubeGame(sys.argv[1])
