#!/usr/bin/env python3

import json
import os
import sys


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
            f"final_hand={self.final_hand}"
        )

    def __parse_game(self):
        for entry in self.__raw_data:
            if entry["type"] == "yourAuthenticationSucceeded":
                self.position = entry["position"]
            if entry["type"] == "youGotCards":
                self.cards = entry["cards"]
            if entry["type"] == "playsTheGame":
                self.game = entry["gameType"]
            if entry["type"] == "gameResult":
                self.result = entry["won"]
                self.points = entry["points"]
                self.skat_cards = entry["skatCards"]
                self.dropped_cards = entry["droppedCards"]
                self.final_hand = self.cards.copy()
                self.final_hand = self.final_hand + self.skat_cards
                for card in self.dropped_cards:
                    self.final_hand.remove(card)


game_type = {
    "Karo": 0,
    "Herz": 1,
    "Pik": 2,
    "Kreuz": 3,
}

card_index = {
    "S7": (0, 0),
    "S8": (0, 1),
    "S9": (0, 2),
    "SU": (0, 3),
    "SO": (0, 4),
    "SK": (0, 5),
    "SX": (0, 6),
    "SA": (0, 7),
    "H7": (1, 0),
    "H8": (1, 1),
    "H9": (1, 2),
    "HU": (1, 3),
    "HO": (1, 4),
    "HK": (1, 5),
    "HX": (1, 6),
    "HA": (1, 7),
    "G7": (2, 0),
    "G8": (2, 1),
    "G9": (2, 2),
    "GU": (2, 3),
    "GO": (2, 4),
    "GK": (2, 5),
    "GX": (2, 6),
    "GA": (2, 7),
    "E7": (3, 0),
    "E8": (3, 1),
    "E9": (3, 2),
    "EU": (3, 3),
    "EO": (3, 4),
    "EK": (3, 5),
    "EX": (3, 6),
    "EA": (3, 7),
}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("provide 1 arg, the filename")
        exit(1)
    parser = SkatstubeGame(sys.argv[1])
