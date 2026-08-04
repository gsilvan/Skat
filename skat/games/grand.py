from typing import Optional

import numpy as np

import skat.games
from skat.card import RANKS, SUITS, Card
from skat.hand import HandOrder
from skat.trick import Trick

#: Grand's only trumps are the four jacks, and the eight plain cards of each
#: suit. Built once at import for the same reason as in `skat.games.suit` -
#: `Trick.forced_cards` and `winner` ask for them constantly during search.
TRUMP_CARDS: tuple[Card, ...] = tuple(
    Card(suit, RANKS.index("J")) for suit in range(len(SUITS))
)

SUIT_CARDS: tuple[tuple[Card, ...], ...] = tuple(
    tuple(Card(suit, rank) for rank in range(len(RANKS)) if rank != RANKS.index("J"))
    for suit in range(len(SUITS))
)


class Grand(skat.games.Game):
    def __init__(self) -> None:
        self.trick = GrandTrick()
        self.order = HandOrder(pivot_ranks="J")

    def new_trick(self) -> None:
        self.trick = GrandTrick()

    @staticmethod
    def trump_cards(suit=None) -> tuple[Card, ...]:
        return TRUMP_CARDS

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        return SUIT_CARDS[suit]

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
                return set(TRUMP_CARDS)
            else:
                return set(SUIT_CARDS[SUITS.index(self.buffer[0].card.suit)])

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
