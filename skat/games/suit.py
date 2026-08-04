from typing import Optional

import numpy as np

import skat.games
from skat.card import RANKS, SUITS, Card
from skat.hand import HandOrder
from skat.trick import Trick

MULTIPLIERS = (9, 10, 11, 12)

JACK = RANKS.index("J")

#: Trumps per suit, ascending: the suit's own cards without its jack, then all
#: four jacks. Built once at import rather than per call — `Trick.winner` asks
#: for these on every card comparison, and rebuilding the tuple there was the
#: single hottest thing in the MCTS search (a majority of its runtime).
TRUMP_CARDS: tuple[tuple[Card, ...], ...] = tuple(
    tuple(
        [Card(suit, rank) for rank in range(len(RANKS)) if rank != JACK]
        + [Card(jack_suit, JACK) for jack_suit in range(len(SUITS))]
    )
    for suit in range(len(SUITS))
)

#: The same eight cards per suit that are *not* trump in that suit's own game.
SUIT_CARDS: tuple[tuple[Card, ...], ...] = tuple(
    tuple(Card(suit, rank) for rank in range(len(RANKS)) if rank != JACK)
    for suit in range(len(SUITS))
)

#: Membership sets for the above. `card in TRUMPS[suit]` on the tuple is a
#: linear scan of Card.__eq__; on the frozenset it is a hash lookup.
_TRUMP_SET: tuple[frozenset[Card], ...] = tuple(
    frozenset(trumps) for trumps in TRUMP_CARDS
)


class SuitGame(skat.games.Game):
    """Implements rules for a Suit-Game"""

    #: Always set, unlike the base class default - a suit game has a trump suit.
    suit: int
    #: Narrower than the base class `Trick`: a suit game always deals in
    #: SuitGameTricks, which know about trumps.
    trick: "SuitGameTrick"

    def __init__(self, suit):
        self.suit = suit
        self.trick = SuitGameTrick(self.suit)
        self.order = HandOrder(suits=self.__suit_order(suit), pivot_ranks="J")

    def new_trick(self):
        self.trick = SuitGameTrick(self.suit)

    @staticmethod
    def trump_cards(suit=None) -> tuple[Card, ...]:
        # TODO: option for ascending, descending
        """
        Returns an ascending ordered tuple of trumps for the selected game
        """
        return TRUMP_CARDS[suit]

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        # TODO: option for ascending, descending
        return SUIT_CARDS[suit]

    @property
    def value(self) -> int:
        """Returns the base multiplier for the selected game"""
        return MULTIPLIERS[self.suit]

    def to_numpy(self) -> np.ndarray:
        arr_size = 5
        arr = np.zeros(arr_size, dtype=int)
        for i in range(arr_size - 1):
            # encode color
            if i == self.suit:
                arr[i] = 1
        # encode current trick value TODO: check if current or interpolated works better
        arr[4] = self.trick.value
        return arr

    @staticmethod
    def __suit_order(suit: int) -> str:
        match suit:
            case 0:
                return "♥♠♣♦"
            case 1:
                return "♦♠♣♥"
            case 2:
                return "♦♥♣♠"
            case 3:
                return "♦♥♠♣"
            case _:
                raise Exception("0-3 is valid, nothing else!")


class SuitGameTrick(Trick):
    def __init__(self, suit):
        super().__init__()
        self.trump_suit = suit

    def is_trump_card(self, card: Card) -> bool:
        """Check if given card is a trump card"""
        return card in _TRUMP_SET[self.trump_suit]

    def is_trump_in_trick(self) -> bool:
        trumps = _TRUMP_SET[self.trump_suit]
        return any(card in trumps for _, card in self.buffer)

    @property
    def forced_cards(self) -> set[Card]:
        if len(self.buffer) == 0:
            return set()
        else:
            if self.is_trump:
                return set(TRUMP_CARDS[self.trump_suit])
            else:
                return set(SUIT_CARDS[SUITS.index(self.buffer[0].card.suit)])

    @property
    def is_trump(self) -> bool:
        if len(self.buffer) <= 0:
            raise Exception("No card in trick")
        return self.is_trump_card(self.buffer[0].card)

    @property
    def winner(self) -> Optional[int]:
        """Returns the trick winner's player_id"""
        if not self.is_full:
            return None
        leading = self.buffer[0]
        if self.is_trump_in_trick():
            # then only here can be the winner
            for turn in self.buffer[1:]:
                if self.is_trump_card(leading.card):
                    if self.is_trump_card(turn.card):
                        if leading.card.is_jack:
                            if turn.card.is_jack:
                                if leading.card < turn.card:
                                    leading = turn
                                else:
                                    continue
                            else:
                                continue
                        else:
                            if turn.card.is_jack:
                                leading = turn
                            else:
                                if leading.card < turn.card:
                                    leading = turn
                    else:
                        continue
                else:
                    if self.is_trump_card(turn.card):
                        leading = turn
            return leading.player_id
        else:
            for turn in self.buffer[1:]:
                if turn.card.suit == self.buffer[0].card.suit:
                    if turn.card > leading.card:
                        leading = turn
            return leading.player_id
