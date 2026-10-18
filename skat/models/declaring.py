"""Small card transformer scoring a proposed pickup declaration."""

import torch
from torch import nn

from skat.agents.neural_declaring.features import INPUT_SIZE


class DeclarationNet(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )
        self.register_buffer("suits", torch.arange(32) // 8, persistent=False)
        self.register_buffer("ranks", torch.arange(32) % 8, persistent=False)
        self.suit = nn.Embedding(4, d_model)
        self.rank = nn.Embedding(8, d_model)
        self.membership = nn.Linear(3, d_model)
        self.context = nn.Linear(9, d_model)
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
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, state):
        if state.ndim != 2 or state.shape[1] != INPUT_SIZE:
            raise ValueError("expected (batch, 105) declaration candidates")
        roles = state[:, :96].reshape(-1, 3, 32).transpose(1, 2)
        context = state[:, 96:]
        tokens = (
            self.suit(self.suits)
            + self.rank(self.ranks)
            + self.membership(roles)
            + self.context(context).unsqueeze(1)
        )
        encoded = self.encoder(tokens)
        known = roles[:, :, 0] + roles[:, :, 1]
        pooled = (encoded * known.unsqueeze(-1)).sum(1) / known.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)
