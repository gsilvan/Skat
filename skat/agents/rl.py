import random
from typing import Optional

import skat.models
from skat.agents import Agent
from skat.card import Card
from skat.games import Game
from skat.player import Player


class RlAgent(Agent):
    def __init__(self):
        self.state: Optional[Player] = None
        self.model = None

    def choose_card(self, valid_moves: set[Card]) -> Card:
        if self.model is None:
            self.model = self.__create_model(
                phase=None, declared_game=None, solo_player=True
            )
        return random.choice(list(valid_moves))  # TODO: select card from model

    def pickup_skat(self, state) -> bool:
        pass

    def bid(self, current_bid, offer=False) -> int:
        pass

    def declare_game(self, state) -> Game:
        pass

    def press_skat(self) -> list[Card]:
        pass

    @staticmethod
    def __create_model(phase, declared_game=None, solo_player=False):
        """Factory method, returns a Suitable Net for a given game."""
        return skat.models.SuitSoloNet()
