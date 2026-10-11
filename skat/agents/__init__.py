"""Strategies select actions; only the engine changes cards and game state."""

from abc import ABC, abstractmethod
from random import Random

from skat.agents.bidding import VALID_BIDS, best_game, next_rung, playable


class Agent(ABC):
    VALID_BIDS = VALID_BIDS

    def __init__(self, seed=None):
        self.state = None  # PlayerView, never Player or Round
        self.rng = Random(seed)
        self._ceiling = None

    def set_state(self, state):
        self.state = state

    def intended_game(self):
        return best_game(set(self.state.hand))

    def max_bid(self):
        if self._ceiling is None:
            cards = set(self.state.hand)
            self._ceiling = (
                self.intended_game().spielwert_range(cards)[0] if playable(cards) else 0
            )
        return self._ceiling

    def bid(self, current_bid, offer=None):
        if offer is not None:
            return offer if offer <= self.max_bid() else 0
        return next_rung(current_bid, self.max_bid())

    def new_round_event(self):
        self._ceiling = None

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
