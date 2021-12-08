from abc import ABC, abstractmethod

from skat.card import Card
from skat.games import Game
from skat.player import Player


class Agent(ABC):
    """Defines a minimal Agent Object"""
    @abstractmethod
    def choose_card(self, state) -> Card:
        raise NotImplementedError

    @abstractmethod
    def pickup_skat(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def bid(self, current_bid) -> int:
        raise NotImplementedError

    @abstractmethod
    def declare_game(self, state) -> Game:
        raise NotImplementedError

    @abstractmethod
    def press_skat(self) -> list[Card]:
        raise NotImplementedError

    def set_state(self, state: Player) -> None:
        raise NotImplementedError
