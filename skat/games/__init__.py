from abc import ABC, abstractmethod

from skat.card import Card
from skat.trick import Trick


class Game(ABC):
    trick: Trick

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def new_trick(self) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trump_cards(suit) -> tuple[Card, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError
