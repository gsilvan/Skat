from abc import ABC, abstractmethod
from typing import Optional

from skat.agents.bidding import (
    VALID_BIDS,
    BiddingStyle,
    best_game,
    ceiling,
    next_rung,
)
from skat.card import Card
from skat.games import Game
from skat.player import Player


class Agent(ABC):
    """Defines a minimal Agent Object"""

    state: Optional[Player]

    VALID_BIDS = VALID_BIDS

    def __init__(self, bidding: BiddingStyle = BiddingStyle.CONSERVATIVE) -> None:
        self.state = None
        self.bidding = bidding
        self._ceiling: Optional[int] = None

    @abstractmethod
    def choose_card(self, valid_moves: set[Card]) -> Card:
        raise NotImplementedError

    @abstractmethod
    def pickup_skat(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def declare_game(self, state) -> Game:
        raise NotImplementedError

    @abstractmethod
    def press_skat(self) -> list[Card]:
        raise NotImplementedError

    # -------------------------------------------------------------- bidding

    def intended_game(self) -> Game:
        """
        The game this agent expects to declare, used to value the hand while
        bidding. It has to agree with `declare_game`, or a conservative bidder
        can still end up überreizt by declaring something cheaper than what it
        priced its bid on.
        """
        if self.state is None:
            raise Exception("Can't value a hand without having a state.")
        return best_game(set(self.state.hand))

    def max_bid(self) -> int:
        """
        Highest value this agent will say, per its `BiddingStyle`. Cached for
        the round - the hand does not change during an auction - and cleared by
        `new_round_event`.
        """
        if self._ceiling is None:
            if self.state is None:
                raise Exception("Can't value a hand without having a state.")
            self._ceiling = ceiling(
                self.bidding, self.intended_game(), set(self.state.hand)
            )
        return self._ceiling

    def bid(self, current_bid, offer: Optional[int] = None) -> int:
        """
        Take part in the Sagen/Hören bidding duel (ISkO 3.3.2). The mechanics
        are the same for every agent; only `max_bid` differs.

        Called as the "sayer" (`offer=None`): return the next higher value from
        `VALID_BIDS` we can still afford, or `0` to pass.

        Called as the "hearer" (`offer=<value just said>`): return `offer` to
        hold that exact value, or `0` to pass.
        """
        limit = self.max_bid()
        if offer:
            return offer if offer <= limit else 0
        return next_rung(current_bid, limit)

    # --------------------------------------------------------------- events

    def set_state(self, state: Player) -> None:
        self.state = state

    def new_round_event(self) -> None:
        """Called before dealing, for agents holding per-round state."""
        self._ceiling = None  # a new hand needs a new valuation

    def trick_done_event(self, is_terminal) -> None:
        return
