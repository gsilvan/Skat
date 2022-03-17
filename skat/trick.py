from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional

import numpy as np

from skat.card import Card

Turn = namedtuple("Turn", ("player_id", "card"))


class Trick(ABC):
    def __init__(self) -> None:
        self.buffer: list[Turn] = list()

    def __len__(self) -> int:
        return len(self.buffer)

    def __str__(self) -> str:
        return str(self.buffer)

    def append(self, player_id: int, card: Card) -> None:
        """Adds a (player, card)-tuple to current trick"""
        if len(self.buffer) < 3:
            self.buffer.append(Turn(player_id, card))
        else:
            raise Exception("can't add more than 3 cards for a single trick")
            # TODO: Use a more specific Exception.

    @property
    def is_full(self) -> bool:
        """
        Returns true if the trick is full, meaning all players placed their
        card. Returns false if cards missing."""
        return len(self.buffer) == 3

    @property
    def forced_cards(self) -> set[Card]:
        raise NotImplementedError

    @property
    def is_trump(self) -> bool:
        """Returns True if first card in Trick is a trump card"""
        raise NotImplementedError

    @property
    def value(self) -> int:
        """Returns the {current, final} value of a trick"""
        _sum = 0
        for _, card in self.buffer:
            _sum += card.value
        return _sum

    @property
    @abstractmethod
    def winner(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def as_vector(self) -> np.ndarray:
        """One-hot-encoded representation of the trick."""
        arr = np.zeros(32)
        for player_id, card in self.buffer:
            arr[card.np_index] = 1
        return arr


class TrickHistory:
    def __init__(self) -> None:
        self.buffer: list[Trick] = list()

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, trick: Trick):
        self.buffer.append(trick)

    def to_numpy(self, player_id=None) -> np.ndarray:
        """
        Return the trick history as one-hot-encoed vector. Either for all player, or
        for a specific player (defined by player_id) only.
        """
        arr = np.zeros(32)
        for trick in self.buffer:
            for pid, card in trick.buffer:
                if player_id is None:
                    arr[card.np_index] = 1
                else:
                    if pid == player_id:
                        arr[card.np_index] = 1
        return arr
