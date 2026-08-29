from __future__ import annotations

import torch
from torch import nn


class BiddingNet(nn.Module):
    """Estimate whether a ten-card hand can win as soloist.

    The first 32 inputs are the cards.  The last three are the bidder's
    relative position: front hand, middle hand, or rear hand.  The single
    output is a logit; its sigmoid is the estimated probability of winning.
    """

    CARDS = 32
    POSITIONS = 3
    INPUT_SIZE = CARDS + POSITIONS

    def __init__(self, hidden_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state).squeeze(-1)
