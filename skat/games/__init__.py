from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

import numpy as np

from skat.card import Card
from skat.hand import HandOrder
from skat.trick import Trick


class Game(ABC):
    trick: Trick
    order: HandOrder = HandOrder()
    #: Trump suit, for the games that have one. `None` for grand and null.
    suit: Optional[int] = None

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

    def matadors(self, cards: Iterable[Card]) -> int:
        """
        Length of the unbroken run of top trumps `cards` holds ("mit N") or
        lacks ("ohne N"), counted from ♣J downwards (ISkO 4.1.2).
        """
        trumps = tuple(reversed(self.trump_cards(self.suit)))
        if not trumps:
            return 0
        cards = set(cards)
        holding = trumps[0] in cards
        count = 0
        for card in trumps:
            if (card in cards) != holding:
                break
            count += 1
        return count

    def spielwert(self, cards: Iterable[Card]) -> int:
        """
        What this game is worth to a declarer holding `cards` — their ten cards
        *plus the skat*, which counts towards the matadors either way and is
        what makes a hand game a gamble (ISkO 4.1.2).
        """
        return (self.matadors(cards) + 1) * self.value

    def spielwert_range(self, cards: Iterable[Card]) -> tuple[int, int]:
        """
        Lowest and highest this game could turn out to be worth once the two
        skat cards join `cards` — the spread a bidder is gambling on.

        The two directions are not symmetric. A "mit" hand can only gain (the
        run of held top trumps can only grow), while an "ohne" hand can only
        lose: a single top matador appearing in the skat cuts the run of
        missing trumps short. That asymmetry is how a soloist ends up
        überreizt, and why the low end is the only value that is safe to bid.

        Only a trump the hand lacks can move the matadors, and at most two can
        arrive, so trying every pair of missing trumps covers every skat.
        """
        held = set(cards)
        missing = [card for card in self.trump_cards(self.suit) if card not in held]
        values = [self.spielwert(held)]
        for i, first in enumerate(missing):
            values.append(self.spielwert(held | {first}))
            for second in missing[i + 1 :]:
                values.append(self.spielwert(held | {first, second}))
        return min(values), max(values)

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError
