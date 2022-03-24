from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from skat.agents import Agent
from skat.card import Card
from skat.games.grand import Grand
from skat.games.null import Null
from skat.games.suit import SuitGame
from skat.player import Player

if TYPE_CHECKING:
    from skat.games import Game


class CommandLineAgent(Agent):
    def __init__(self) -> None:
        self.state: Optional[Player] = None

    def choose_card(self, valid_moves: set[Card]) -> Card:
        print(f"choose one of {list(valid_moves)}:")
        index = None
        while index is None:
            user_input = input("index: ")
            try:
                _in = int(user_input)
                if _in in range(0, len(valid_moves)):
                    index = _in
            except ValueError:
                print("Error: index out of range!")
        choice = list(valid_moves)[index]
        self.state.hand.remove(choice)
        return choice

    def pickup_skat(self, state) -> bool:
        user_input = input("pick up the skat? [y/n]: ")  # TODO: input sanitation
        if user_input == "y":
            return True
        elif user_input == "n":
            return False
        else:
            return True

    def bid(self, current_bid, offer=False) -> int:
        print(f"current bid: {current_bid}")
        user_input = input("your bid: ")  # TODO: input sanitation
        return int(user_input)

    def declare_game(self, state) -> Game:
        available_games = ["Diamonds", "Hearts", "Spades", "Clubs", "Grand", "Null"]
        user_input = int(input(f"select one of: {available_games}"))
        # TODO: input sanitation
        match user_input:
            case 0:
                return SuitGame(0)
            case 1:
                return SuitGame(1)
            case 2:
                return SuitGame(2)
            case 3:
                return SuitGame(3)
            case 4:
                return Grand()
            case 5:
                return Null()
            case _:
                raise Exception("choose 0..5. nothing else!")

    def press_skat(self) -> list[Card]:
        if self.state is None:
            raise Exception("a state is missing")
        skat = list()
        for _ in range(2):
            user_input = input(f"select one of {self.state.hand}")  # TODO: input sanit.
            skat.append(self.state.hand.pop(int(user_input)))
        return skat
