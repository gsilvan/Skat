from abc import ABC, abstractmethod

import numpy as np

from skat.card import Card
from skat.hand import HandOrder
from skat.trick import Trick


class Game(ABC):
    trick: Trick
    order: HandOrder = HandOrder()

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def new_trick(self) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trump_cards(suit=None) -> tuple[Card, ...]:
        raise NotImplementedError

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError
