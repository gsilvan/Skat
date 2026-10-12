"""Strategies select actions; only the engine changes cards and game state."""

from abc import ABC, abstractmethod
from random import Random

from skat.agents.bidding import VALID_BIDS, BiddingStyle, best_game, next_rung, playable


class Agent(ABC):
    VALID_BIDS = VALID_BIDS

    def __init__(self, seed=None):
        self.state = None  # PlayerView, never Player or Round
        self.rng = Random(seed)
        self._ceiling = None
        self.bidding = BiddingStyle.CONSERVATIVE
        self.neural_bidder = None
        self.last_bidding_probability = None

    def set_state(self, state):
        self.state = state

    @property
    def bidding_label(self):
        if self.bidding is BiddingStyle.NEURAL and self.neural_bidder is not None:
            return f"neural ({self.neural_bidder.threshold:g})"
        return self.bidding.value

    def intended_game(self):
        return best_game(set(self.state.hand))

    def max_bid(self):
        if self._ceiling is None:
            cards = set(self.state.hand)
            if self.bidding is BiddingStyle.NEURAL:
                if self.neural_bidder is None:
                    raise ValueError("neural bidding requires a bidding model")
                self.last_bidding_probability = self.neural_bidder.probability(
                    cards, self.state.seat_id, self.state.dealer
                )
                accepted = self.last_bidding_probability >= self.neural_bidder.threshold
            else:
                accepted = playable(cards)
            self._ceiling = (
                self.intended_game().spielwert_range(cards)[0] if accepted else 0
            )
        return self._ceiling

    def configure_bidding(self, style="conservative", bidder=None):
        style = BiddingStyle(style)
        if style is BiddingStyle.NEURAL and bidder is None:
            raise ValueError("neural bidding requires a bidding model")
        self.bidding, self.neural_bidder = style, bidder
        self._ceiling = None
        self.last_bidding_probability = None
        return self

    def bid(self, current_bid, offer=None):
        if offer is not None:
            return offer if offer <= self.max_bid() else 0
        return next_rung(current_bid, self.max_bid())

    def new_round_event(self):
        self._ceiling = None
        self.last_bidding_probability = None

    def trick_done_event(self, is_terminal):
        pass

    def confirm_concession(self):
        return False

    def confirm_claim(self, level):
        return False

    @abstractmethod
    def choose_card(self, valid_moves):
        raise NotImplementedError

    @abstractmethod
    def pickup_skat(self, state=None):
        raise NotImplementedError

    @abstractmethod
    def press_skat(self):
        raise NotImplementedError

    @abstractmethod
    def declare_game(self, state=None):
        raise NotImplementedError
