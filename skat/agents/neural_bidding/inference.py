from __future__ import annotations

import os

import torch

from skat.agents.neural_bidding.checkpoint import TRAIN_HINT, load_bidding
from skat.agents.neural_bidding.features import bidding_state
from skat.agents.rl.dqn import describe_device, resolve_device
from skat.card import Card


class NeuralBidder:
    """Shared, read-only bidding model that can be attached to any agent."""

    def __init__(self, path: str, device: str = "auto", threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("bidding threshold must be between 0 and 1")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} does not exist; neural bidding cannot run without a "
                f"trained checkpoint. Train one with:\n  {TRAIN_HINT}"
            )
        self.device = resolve_device(device)
        self.net, self.meta = load_bidding(path, self.device)
        self.net.eval()
        self.threshold = threshold
        print(f"NeuralBidder: {path} on {describe_device(self.device)}")

    def probability(
        self, cards: list[Card] | set[Card], seat: int, dealer: int
    ) -> float:
        state = torch.from_numpy(bidding_state(cards, seat, dealer)).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.net(state.to(self.device))).item())

    def should_bid(self, cards: list[Card] | set[Card], seat: int, dealer: int) -> bool:
        return self.probability(cards, seat, dealer) >= self.threshold
