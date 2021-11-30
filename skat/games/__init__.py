from abc import ABC, abstractmethod

from skat.card import Card


class Game(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def trump_set(self) -> tuple[Card, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def find_winner(self, trick) -> int:
        raise NotImplementedError
