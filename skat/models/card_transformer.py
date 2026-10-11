"""One transformer policy: 32 card tokens and public game context."""

import torch
from torch import nn

from skat.models.observation import INPUT_SIZE, OUTPUT_SIZE


class CardTransformer(nn.Module):
    INPUT_SIZE = INPUT_SIZE
    OUTPUT_SIZE = OUTPUT_SIZE
    STATUS_BLOCKS = (0, 44, 76, 108, 140, 172, 204)

    def __init__(
        self, d_model=128, nhead=4, dim_feedforward=256, num_layers=4, dropout=0.1
    ):
        super().__init__()
        self.config = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
        )
        cards = torch.arange(32)
        self.register_buffer(
            "status_index",
            (torch.tensor(self.STATUS_BLOCKS).view(-1, 1) + cards.view(1, -1)).reshape(
                -1
            ),
            persistent=False,
        )
        self.register_buffer("suit_of", cards // 8, persistent=False)
        self.register_buffer("rank_of", cards % 8, persistent=False)
        self.register_buffer(
            "suit_onehot", torch.eye(4).repeat_interleave(8, 0), persistent=False
        )
        self.register_buffer("is_jack", (cards % 8 == 3).float(), persistent=False)
        self.token = nn.Linear(17, d_model)
        self.suit_embedding = nn.Embedding(4, d_model)
        self.rank_embedding = nn.Embedding(8, d_model)
        self.context = nn.Linear(25, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, 1)

    def features_of(self, x):
        if x.ndim != 2 or x.shape[-1] != self.INPUT_SIZE:
            raise ValueError(f"expected (batch, {self.INPUT_SIZE}) observation")
        batch = len(x)
        status = x[:, self.status_index].reshape(batch, 7, 32).transpose(1, 2)
        unseen = 1 - status.sum(-1, keepdim=True)
        game = x[:, 32:38]
        trump = (1 - game[:, 5:6]) * (
            self.is_jack + game[:, :4] @ self.suit_onehot.t()
        ).clamp(max=1)
        current = x[:, 140:236].reshape(batch, 3, 32)
        position = x[:, 236:239].argmax(-1)
        slots = (
            (torch.arange(3, device=x.device).view(1, 3) + position.view(batch, 1)) % 3
        ).unsqueeze(-1)
        order_matrix = x.new_zeros(batch, 3, 3).scatter_(2, slots, 1)
        order = torch.einsum("brc,bro->bco", current, order_matrix)
        led = order[:, :, 0]
        same_suit = (led @ self.suit_onehot) @ self.suit_onehot.t()
        exposed = x[:, 245:341].reshape(batch, 3, 32).transpose(1, 2)
        skat = x[:, 344:376].unsqueeze(-1)
        tokens = torch.cat(
            (
                status,
                unseen,
                trump.unsqueeze(-1),
                order,
                same_suit.unsqueeze(-1),
                exposed,
                skat,
            ),
            dim=-1,
        )
        context = torch.cat(
            (
                x[:, 32:44],
                x[:, 236:245],
                x[:, 341:344],
                (led * trump).sum(-1, keepdim=True),
            ),
            dim=-1,
        )
        return tokens, context

    def forward(self, x):
        tokens, context = self.features_of(x)
        identity = self.suit_embedding(self.suit_of) + self.rank_embedding(self.rank_of)
        hidden = (
            self.token(tokens)
            + identity.unsqueeze(0)
            + self.context(context).unsqueeze(1)
        )
        return self.head(self.encoder(hidden)).squeeze(-1)
