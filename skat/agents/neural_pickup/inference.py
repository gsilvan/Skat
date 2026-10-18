"""Choose Hand only when its predicted win chance is higher."""

import math

import torch

from skat.agents.neural_pickup.checkpoint import load_pickup
from skat.agents.neural_pickup.features import pickup_state
from skat.utils.reproducibility import resolve_device


class PickupScorer:
    def __init__(self, path, *, device="cpu"):
        self.device = resolve_device(device)
        self.model, self.temperatures, self.metadata = load_pickup(path, self.device)

    def probabilities(self, state):
        features = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(features)[0]
            return tuple(
                float(torch.sigmoid(logit / temperature))
                for logit, temperature in zip(logits, self.temperatures, strict=True)
            )


class NeuralPickup:
    def __init__(self, scorer, *, margin=0.0):
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("pickup margin must be nonnegative and finite")
        self.scorer = scorer
        self.margin = margin

    def choose_pickup(self, view):
        # Existing declaration strategies choose this game after a Hand decision.
        # A hidden skat can change matadors; use the safe lower bound.
        from skat.agents.declaring import HeuristicDeclarer

        cards = set(view.hand)
        hand_game = HeuristicDeclarer().choose_hand(view)
        if hand_game.spielwert_range(cards)[0] < view.highest_bid:
            return True
        pickup, hand = self.scorer.probabilities(
            pickup_state(cards, view.highest_bid, view.seat_id, view.dealer)
        )
        if not all(
            math.isfinite(value) and 0 <= value <= 1 for value in (pickup, hand)
        ):
            raise ValueError("invalid pickup model probabilities")
        return hand <= pickup + self.margin
