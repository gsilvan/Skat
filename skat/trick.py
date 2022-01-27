from abc import ABC, abstractmethod
from typing import Union

from skat.card import Card


class Trick(ABC):
    def __init__(self) -> None:
        self.card_turn: list[tuple[int, Card]] = list()

    def __len__(self) -> int:
        return len(self.card_turn)

    def append(self, player_id: int, card: Card) -> None:
        """Adds a (player, card)-tuple to current trick"""
        if len(self.card_turn) < 3:
            self.card_turn.append((player_id, card))
        else:
            raise Exception("can't add more than 3 cards for a single trick")
            # TODO: Use a more specific Exception.

    @property
    def is_full(self) -> bool:
        """
        Returns true if the trick is full, meaning all players placed their
        card. Returns false if cards missing."""
        return len(self.card_turn) == 3

    @property
    def value(self) -> int:
        """Returns the {current, final} value of a trick"""
        _sum = 0
        for _, card in self.card_turn:
            _sum += card.value
        return _sum

    @property
    @abstractmethod
    def winner(self) -> Union[int, None]:
        raise NotImplementedError
