from __future__ import annotations

import torch

#: Visit counts are already a distribution over moves; nothing to sharpen by
#: default.
DEFAULT_VISIT_TEMPERATURE = 1.0

#: Mean values sit in `PlayoutState.rewards()`'s [0, 1] and differ between good
#: and bad moves by a few hundredths, so a plain softmax over them would be
#: almost uniform. This is the scale that turns them into a usable target.
DEFAULT_VALUE_TEMPERATURE = 0.05


def visit_targets(
    visits: torch.Tensor,
    masks: torch.Tensor,
    temperature: float = DEFAULT_VISIT_TEMPERATURE,
) -> torch.Tensor:
    """
    MCTS visit counts as a distribution over the legal cards: zero outside the
    mask, each row summing to 1.

    `temperature < 1` sharpens towards the most-visited move, `> 1` flattens.
    The counts are stored raw on disk precisely so this stays a knob on the
    training run rather than a property of the corpus.
    """
    counts = visits.to(torch.float32).masked_fill(~masks, 0.0)
    if temperature != 1.0:
        counts = counts.pow(1.0 / temperature)
    return counts / counts.sum(-1, keepdim=True)


def value_targets(
    values: torch.Tensor,
    visits: torch.Tensor,
    masks: torch.Tensor,
    temperature: float = DEFAULT_VALUE_TEMPERATURE,
    bonuses: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    A softmax over each move's *mean* value - the quantity `MCTSAgent.best`
    actually maximises.

    Worth having next to `visit_targets`, because the visit counts turn out to
    be a *flat* target: measured over a 2000-simulation corpus they average a
    peak probability of 0.31 against 0.21 for uniform over 4.7 legal moves, and
    their argmax is the card the teacher actually played only 81% of the time.
    UCB1 spends most of its budget exploring, so how often the search looked at
    a move says much less than how the move scored. At the same temperature the
    value target peaks at 0.47 and agrees with the played card 99% of the time.

    That said, the difference does *not* show up in playing strength: trained
    on the same 102k-decision corpus the two targets came out level over 2000
    seeded deals (80.4% vs 81.6% of rounds won, about one standard error). So
    `visits` stays the default and this is the documented alternative, not the
    upgrade the sharpness numbers suggest it would be.

    A move the search never visited has no value to speak of, so it is dropped
    from the softmax rather than entered with a 0.
    """
    visited = masks & (visits > 0)
    mean = torch.where(visited, values / visits.clamp(min=1), values.new_zeros(()))
    if bonuses is not None:
        # Teacher ties refer to exact primary means, without an epsilon. Use
        # float64 for the comparison so float32 division cannot merge two
        # distinct empirical win rates. Bonuses only raise tied maxima; a
        # lower primary mean can never become the target's preferred move.
        precise = values.double() / visits.double().clamp(min=1)
        precise = precise.masked_fill(~visited, -torch.inf)
        top = precise == precise.max(-1, keepdim=True).values
        tied = top & (top.sum(-1, keepdim=True) > 1)
        bonus_mean = bonuses.double() / visits.double().clamp(min=1)
        low = bonus_mean.masked_fill(~top, torch.inf).min(-1, keepdim=True).values
        mean = precise + temperature * torch.where(tied, bonus_mean - low, 0.0)
    logits = (mean / temperature).masked_fill(~visited, torch.finfo(mean.dtype).min)
    return torch.softmax(logits, dim=-1).to(values.dtype)
