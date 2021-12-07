from abc import ABC, abstractmethod

from skat.card import Card
from skat.games import Game


class Agent(ABC):
    """Defines a minimal Agent Object"""
    @abstractmethod
    def choose_card(self, state) -> Card:
        raise NotImplementedError

    @abstractmethod
    def pickup_skat(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def bid(self, state) -> int:
        raise NotImplementedError

    @abstractmethod
    def declare_game(self, state) -> Game:
        raise NotImplementedError
