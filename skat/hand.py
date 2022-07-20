import collections.abc
import typing
from typing import NamedTuple

import numpy as np
import torch

from skat.card import Card

FULL_HAND = [Card(i, j) for i in range(4) for j in range(8)]


class HandOrder(NamedTuple):
    suits: str = "♦♥♠♣"
    ranks: str = ""
    pivot_suits: str = "♦♥♠♣"
    pivot_ranks: str = ""


class Hand(collections.abc.MutableSequence):
    def __init__(self, iterable=None) -> None:
        self.__hand: list[Card] = list()
        if iterable:
            for item in iterable:
                self.__hand.append(item)

    @typing.no_type_check
    def __setitem__(self, index: int, card: Card) -> None:
        self.__hand[index] = card

    @typing.no_type_check
    def __delitem__(self, index: int) -> None:
        del self.__hand[index]

    @typing.no_type_check
    def __getitem__(self, index: int):
        return self.__hand[index]

    # TODO: improve typing

    def __len__(self) -> int:
        return len(self.__hand)

    def __str__(self):
        return str(self.__hand)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.__hand == other.__hand

    def append(self, card: Card) -> None:
        self.__hand.append(card)

    def insert(self, index: int, card: Card) -> None:
        self.__hand.insert(index, card)

    @property
    def value(self) -> int:
        """Returns the sum of point values in the hand."""
        _sum = 0
        for card in self.__hand:
            _sum += card.value
        return _sum

    @property
    def as_vector(self) -> np.ndarray:
        arr = np.zeros(32)
        for card in self.__hand:
            arr[card.np_index] = 1
        return arr

    @property
    def as_tensor_mask(self) -> torch.Tensor:
        tensor = torch.zeros(32, dtype=bool, device="cuda")
        # TODO: device should be derived from a config file
        for card in self.__hand:
            tensor[card.np_index] = True
        return tensor

    def sort(self, order: HandOrder):
        """Sort the Hand according to specified sort rules using Bubble-Sort."""
        while True:
            has_changed = False
            for i in range(len(self.__hand) - 1):
                if not self.tgt(self.__hand[i], self.__hand[i + 1], order):
                    pivot = self.__hand[i]
                    self.__hand[i] = self.__hand[i + 1]
                    self.__hand[i + 1] = pivot
                    has_changed = True
            if not has_changed:
                return

    @staticmethod
    def tgt(a: Card, b: Card, order: HandOrder):
        """Test if card `a` is better than card `b`"""
        pivot_chars = tuple(order.pivot_ranks)
        pivot_suits = tuple(order.pivot_suits)
        if a.rank in pivot_chars or b.rank in pivot_chars:
            if a.rank == b.rank:
                return pivot_suits.index(a.suit) > pivot_suits.index(b.suit)
            if a.rank in pivot_chars and b.rank in pivot_chars:
                return pivot_chars.index(a.rank) > pivot_chars.index(b.rank)
            else:
                return a.rank in pivot_chars
        if a.suit == b.suit:
            return tuple(order.ranks).index(a.rank) > tuple(order.ranks).index(b.rank)
        if not a.suit == b.suit:
            return tuple(order.suits).index(a.suit) > tuple(order.suits).index(b.suit)
        raise Exception("Something went completely wrong!")
