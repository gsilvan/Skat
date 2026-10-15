"""Card-token transformer for three possible hidden-card locations."""

import torch
from torch import nn

from skat.agents.belief.features import INPUT_SIZE, OBSERVATION_SIZE
from skat.models.card_transformer import CardTransformer


class BeliefTransformer(CardTransformer):
    INPUT_SIZE = INPUT_SIZE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token = nn.Linear(21, self.config["d_model"])
        self.head = nn.Linear(self.config["d_model"], 3)

    def features_of(self, x):
        if x.ndim != 2 or x.shape[-1] != self.INPUT_SIZE:
            raise ValueError(f"expected (batch, {self.INPUT_SIZE}) belief input")
        tokens, context = super().features_of(x[:, :OBSERVATION_SIZE])
        extra = x[:, OBSERVATION_SIZE:].reshape(len(x), 4, 32).transpose(1, 2)
        return torch.cat((tokens, extra), dim=-1), context


def masked_log_probs(logits, inputs):
    """Mask impossible locations before cardwise cross-entropy."""
    voids = inputs[:, OBSERVATION_SIZE : OBSERVATION_SIZE + 64].reshape(-1, 2, 32)
    forbidden = torch.stack(
        (voids[:, 0], voids[:, 1], torch.zeros_like(voids[:, 0])), dim=-1
    )
    return torch.log_softmax(logits.masked_fill(forbidden > 0, -1e9), -1)
