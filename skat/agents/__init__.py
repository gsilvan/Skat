from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import TYPE_CHECKING, Optional

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

if TYPE_CHECKING:
    from skat.agents.neural_bidding import NeuralBidder


class Agent(ABC):
    """Defines a minimal Agent Object"""

    state: Optional[Player]

    VALID_BIDS = VALID_BIDS

    def __init__(
        self,
        bidding: BiddingStyle = BiddingStyle.CONSERVATIVE,
        seed: Optional[int] = None,
    ) -> None:
        self.state = None
        self.bidding = bidding
        self.rng = Random(seed)
        self._ceiling: Optional[int] = None
        self.neural_bidder: Optional[NeuralBidder] = None
        self.last_bidding_probability: Optional[float] = None

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
            cards = set(self.state.hand)
            if self.bidding is BiddingStyle.NEURAL:
                if self.neural_bidder is None:
                    raise RuntimeError(
                        "BiddingStyle.NEURAL requires a NeuralBidder checkpoint."
                    )
                round_ = self.state.public_state
                if round_ is None:
                    raise RuntimeError("neural bidding requires a seated round")
                self.last_bidding_probability = self.neural_bidder.probability(
                    cards, self.state.seat_id, round_.dealer
                )
                if self.last_bidding_probability < self.neural_bidder.threshold:
                    self._ceiling = 0
                else:
                    self._ceiling = ceiling(
                        BiddingStyle.CONSERVATIVE, self.intended_game(), cards
                    )
            else:
                self._ceiling = ceiling(
                    self.bidding,
                    self.intended_game(),
                    cards,
                    rng=self.rng,
                )
        return self._ceiling

    def set_neural_bidder(self, bidder: NeuralBidder) -> None:
        """Attach one immutable bidding model; it may be shared by all seats."""
        self.neural_bidder = bidder

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
        self.last_bidding_probability = None

    def trick_done_event(self, is_terminal) -> None:
        return
