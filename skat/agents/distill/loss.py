from __future__ import annotations

import torch


def masked_log_softmax(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Log-softmax over the legal cards only.

    Out-of-place on purpose. `skat.agents.rl.mask.MaskedCategorical` does the
    same job with `logits.masked_fill_`, which mutates the network's own output
    tensor - harmless for a no-grad argmax, wrong inside a backward pass and
    wrong again for anyone who wanted to read the raw logits afterwards.
    """
    return torch.log_softmax(
        logits.masked_fill(~masks, torch.finfo(logits.dtype).min), dim=-1
    )


def policy_loss(
    logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """
    Cross-entropy H(teacher, policy), summed over the legal cards and averaged
    over the batch.

    The second `masked_fill` keeps an illegal card's `0 * -3.4e38` term out of
    the sum. It is finite today only because the fill value is `finfo.min`
    rather than `-inf`; zeroing the term says what is meant instead of relying
    on that, and it is what makes the loss provably independent of an illegal
    logit rather than merely insensitive to it.
    """
    log_p = masked_log_softmax(logits, masks).masked_fill(~masks, 0.0)
    return -(targets * log_p).sum(-1).mean()
