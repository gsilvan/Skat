from abc import ABC, abstractmethod

from skat.card import Card


class Game(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def trumps(self) -> list[Card]:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate_trick(trick) -> tuple[int, int]:
        raise NotImplementedError
