"""Hand classifier independent of the card-playing network."""

import torch
from torch import nn

from skat.agents.neural_bidding.features import INPUT_SIZE


class BiddingNet(nn.Module):
    INPUT_SIZE = INPUT_SIZE

    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )
        self.register_buffer("suits", torch.arange(32) // 8, persistent=False)
        self.register_buffer("ranks", torch.arange(32) % 8, persistent=False)
        self.suit = nn.Embedding(4, d_model)
        self.rank = nn.Embedding(8, d_model)
        self.membership = nn.Linear(1, d_model)
        self.position = nn.Linear(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            2 * d_model,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, state):
        if state.ndim != 2 or state.shape[-1] != INPUT_SIZE:
            raise ValueError("expected (batch, 35) original-hand observations")
        held = state[:, :32]
        tokens = (
            self.suit(self.suits)
            + self.rank(self.ranks)
            + self.membership(held.unsqueeze(-1))
            + self.position(state[:, 32:]).unsqueeze(1)
        )
        encoded = self.encoder(tokens)
        pooled = (encoded * held.unsqueeze(-1)).sum(1) / held.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)
