import random

import skat.models
from skat.agents import Agent
from skat.agents.bidding import BiddingStyle
from skat.card import Card
from skat.games import Game


class DeepAgent(Agent):
    def __init__(
        self,
        bidding_model=None,
        play_model=None,
        bidding: BiddingStyle = BiddingStyle.CONSERVATIVE,
    ):
        super().__init__(bidding=bidding)
        self.model = play_model

    def choose_card(self, valid_moves: set[Card]) -> Card:
        if self.model is None:
            self.model = self.__create_model(
                phase=None, declared_game=None, solo_player=True
            )
        return random.choice(list(valid_moves))  # TODO: select card from model

    def pickup_skat(self, state) -> bool:
        pass

    def declare_game(self, state) -> Game:
        pass

    def press_skat(self) -> list[Card]:
        pass

    @staticmethod
    def __create_model(phase, declared_game=None, solo_player=False):
        """Factory method, returns a Suitable Net for a given game."""
        return skat.models.QNet()
