import collections.abc
import typing

import numpy as np

from skat.card import Card


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

    def append(self, card: Card) -> None:
        self.__hand.append(card)

    def insert(self, index: int, card: Card) -> None:
        self.__hand.insert(index, card)

    @property
    def as_vector(self) -> np.ndarray:
        arr = np.zeros(32)
        for card in self.__hand:
            arr[card.np_index] = 1
        return arr
