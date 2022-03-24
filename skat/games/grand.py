from typing import Optional

import numpy as np

import skat.games
from skat.card import RANKS, SUITS, Card
from skat.hand import HandOrder
from skat.trick import Trick


class Grand(skat.games.Game):
    def __init__(self) -> None:
        self.trick = GrandTrick()
        self.order = HandOrder(pivot_ranks="J")

    def new_trick(self) -> None:
        self.trick = GrandTrick()

    @staticmethod
    def trump_cards(suit=None) -> tuple[Card, ...]:
        trumps = list()
        for i, _ in enumerate(SUITS):
            trumps.append(Card(i, 3))
        return tuple(trumps)

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        suits = list()
        for i, _ in enumerate(RANKS):
            if i == 3:
                continue  # skip J
            suits.append(Card(suit, i))
        return tuple(suits)

    @property
    def value(self) -> int:
        return 24

    def to_numpy(self) -> np.ndarray:
        arr_size = 5
        arr = np.zeros(arr_size, dtype=int)
        # encode current trick value TODO: check if current or interpolated works better
        arr[4] = self.trick.value
        return arr


class GrandTrick(Trick):
    @property
    def forced_cards(self) -> set[Card]:
        if len(self.buffer) == 0:
            return set()
        else:
            if self.is_trump:
                return set(Grand.trump_cards())
            else:
                return set(Grand.suit_cards(SUITS.index(self.buffer[0].card.suit)))

    @property
    def is_trump(self) -> bool:
        if len(self.buffer) <= 0:
            return False
        return self.buffer[0].card.is_jack

    @property
    def winner(self) -> Optional[int]:
        if not self.is_full:
            return None
        leading = self.buffer[0]
        if self._is_trump_in_trick():
            for turn in self.buffer[1:]:
                if turn.card.is_jack:
                    if leading.card.is_jack:
                        if leading.card < turn.card:
                            leading = turn
                    else:
                        leading = turn
            return leading.player_id
        else:
            for turn in self.buffer[1:]:
                if turn.card.suit == self.buffer[0].card.suit:
                    if turn.card > leading.card:
                        leading = turn
            return leading.player_id

    def _is_trump_in_trick(self) -> bool:
        return any(x.is_jack for _, x in self.buffer)
