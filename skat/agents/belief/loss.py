from __future__ import annotations

import torch
import torch.nn.functional as F

from skat.agents.belief.samples import KNOWN


def belief_loss(logits: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over the unseen cards only, averaged per card.

    `logits` is `(B, 32, 3)` and `locations` is `(B, 32)` holding a location or
    `KNOWN`. Cards the observer can already see carry no question and are
    dropped - `ignore_index` is exactly that, and it is why the label uses a
    sentinel rather than a separate mask column.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        locations.reshape(-1),
        ignore_index=KNOWN,
    )


def belief_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """`(B, 32, 3)` logits to probabilities over the three locations."""
    return torch.softmax(logits, dim=-1)


def capacity_prior(locations: torch.Tensor, slots: int = 3) -> torch.Tensor:
    """
    What uniform determinization already believes, as a distribution.

    `MCTSAgent` deals the unseen cards uniformly subject to how many cards each
    slot still holds, so its implicit belief about any one card is just the
    share of the remaining places belonging to each slot. **That** is the
    baseline a belief network has to beat - not a flat 1/3, which no version of
    the search has ever used, and against which any network would look good.

    The slot sizes are counted from the labels, which is fair rather than
    leaky: the real search knows them exactly too. It can see how many cards
    each opponent is holding and how big the skat is; what it does not know is
    *which* cards are where, and that is the only thing being predicted.
    """
    unseen = locations >= 0
    counts = torch.stack(
        [((locations == slot) & unseen).sum(-1) for slot in range(slots)], dim=-1
    ).float()  # (B, slots)
    share = counts / counts.sum(-1, keepdim=True).clamp(min=1)
    return share.unsqueeze(1).expand(-1, locations.shape[-1], -1)


def cross_entropy_of(
    probabilities: torch.Tensor, locations: torch.Tensor
) -> torch.Tensor:
    """The same score as `belief_loss`, for a distribution given directly."""
    unseen = locations >= 0
    if not unseen.any():
        return torch.zeros((), device=probabilities.device)
    picked = probabilities.gather(-1, locations.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -torch.log(picked[unseen].clamp(min=1e-12)).mean()


def accuracy_of(probabilities: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Share of unseen cards whose most likely location is the true one."""
    unseen = locations >= 0
    if not unseen.any():
        return torch.zeros((), device=probabilities.device)
    return (probabilities.argmax(-1)[unseen] == locations[unseen]).float().mean()
