from typing import Union

from skat.agents import Agent
from skat.card import Card
from skat.games import Game
from skat.player import Player


class CommandLineAgent(Agent):
    def __init__(self) -> None:
        self.state: Union[Player, None] = None

    def choose_card(self, valid_moves: set[Card]) -> Card:
        pass

    def pickup_skat(self, state) -> bool:
        pass

    def bid(self, current_bid) -> int:
        pass

    def declare_game(self, state) -> Game:
        pass

    def press_skat(self) -> list[Card]:
        pass

