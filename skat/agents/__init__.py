"""Strategies select actions; only the engine changes cards and game state."""

from abc import ABC, abstractmethod
from random import Random

from skat.agents.bidding import VALID_BIDS, BiddingStyle, best_game, next_rung, playable
from skat.agents.declaring import HeuristicDeclarer


class Agent(ABC):
    VALID_BIDS = VALID_BIDS

    def __init__(self, seed=None):
        self.state = None  # PlayerView, never Player or Round
        self.rng = Random(seed)
        self._ceiling = None
        self.bidding = BiddingStyle.CONSERVATIVE
        self.neural_bidder = None
        self.last_bidding_probability = None
        self.declaration_strategy = HeuristicDeclarer()
        self.declaration_style = "heuristic"
        self._declaration_choice = None
        self.pickup_strategy = None
        self.pickup_style = "always"

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

    def configure_declaration(self, style="heuristic", model=None):
        if style == "heuristic":
            self.declaration_strategy = HeuristicDeclarer()
        elif style == "neural":
            if model is None:
                raise ValueError("neural declaration requires a declaration model")
            from skat.agents.neural_declaring.inference import NeuralDeclarer

            self.declaration_strategy = NeuralDeclarer(model)
        else:
            raise ValueError(f"unknown declaration style: {style}")
        self.declaration_style = style
        self._declaration_choice = None
        return self

    def configure_pickup(self, style="always", model=None, margin=0.0):
        if style == "always":
            self.pickup_strategy = None
        elif style == "neural":
            if model is None:
                raise ValueError("neural pickup requires a pickup model")
            from skat.agents.neural_pickup.inference import NeuralPickup

            self.pickup_strategy = NeuralPickup(model, margin=margin)
        else:
            raise ValueError(f"unknown pickup style: {style}")
        self.pickup_style = style
        return self

    @property
    def declaration_label(self):
        return self.declaration_style

    def bid(self, current_bid, offer=None):
        if offer is not None:
            return offer if offer <= self.max_bid() else 0
        return next_rung(current_bid, self.max_bid())

    def new_round_event(self):
        self._ceiling = None
        self.last_bidding_probability = None
        self._declaration_choice = None

    def trick_done_event(self, is_terminal):
        pass

    def confirm_concession(self):
        return False

    def confirm_claim(self, level):
        return False

    @abstractmethod
    def choose_card(self, valid_moves):
        raise NotImplementedError

    def pickup_skat(self, state=None):
        if self.pickup_strategy is not None:
            return self.pickup_strategy.choose_pickup(self.state)
        return self.declaration_strategy.choose_pickup(self.state)

    def press_skat(self):
        self._declaration_choice = self.declaration_strategy.choose_after_pickup(
            self.state
        )
        return self._declaration_choice.pressed

    def declare_game(self, state=None):
        if self.state.hand_game:
            return self.declaration_strategy.choose_hand(self.state)
        if self._declaration_choice is None:
            raise ValueError("declare_game requires a prior press_skat decision")
        game = self._declaration_choice.game
        self._declaration_choice = None
        return game
