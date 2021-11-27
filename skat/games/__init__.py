from abc import ABC, abstractmethod

from skat.card import Card


class Game(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def trumps(self) -> list[Card]:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError
