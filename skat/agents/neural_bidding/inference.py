"""Immutable bidder shared by independent card-playing agents."""

import math

import torch

from skat.agents.neural_bidding.checkpoint import load_bidding
from skat.agents.neural_bidding.features import bidding_state
from skat.utils.reproducibility import resolve_device


class NeuralBidder:
    def __init__(self, path, *, device="cpu", threshold=2 / 3):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("bidding threshold must lie between zero and one")
        self.device = resolve_device(device)
        self.model, self.temperature, self.metadata = load_bidding(path, self.device)
        self.threshold = threshold

    def probability(self, cards, seat, dealer):
        state = (
            torch.from_numpy(bidding_state(cards, seat, dealer))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            return float((self.model(state) / self.temperature).sigmoid().item())
