from abc import ABC, abstractmethod

from skat.card import Card


class Game(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def new_trick(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trump_cards(suit) -> tuple[Card, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError
