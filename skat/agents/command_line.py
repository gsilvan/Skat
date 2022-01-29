from typing import Union

from skat.agents import Agent
from skat.card import Card
from skat.games import Game
from skat.games.suit import SuitGame
from skat.player import Player


class CommandLineAgent(Agent):
    def __init__(self) -> None:
        self.state: Union[Player, None] = None

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

    def bid(self, current_bid) -> int:
        print(f"current bid: {current_bid}")
        user_input = input("your bid: ")  # TODO: input sanitation
        return int(user_input)

    def declare_game(self, state) -> Game:
        available_games = ["Diamonds", "Hearts", "Spades", "Clubs"]
        user_input = input(f"select one of: {available_games}")  # TODO: input sanit.
        return SuitGame(int(user_input))

    def press_skat(self) -> list[Card]:
        skat = list()
        for _ in range(2):
            user_input = input(f"select one of {self.state.hand}")  # TODO: input sanit.
            skat.append(self.state.hand.pop(int(user_input)))
        return skat
