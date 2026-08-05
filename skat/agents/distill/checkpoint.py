from __future__ import annotations

import os
from typing import Any

import torch

from skat.models.policy import PolicyNet

#: Written into every policy checkpoint. Its only job is to be absent from a
#: DQN checkpoint, which is a bare state dict.
POLICY_FORMAT = "skat-policy-v1"

TRAIN_HINT = "python -m skat.agents.distill.train --data <dir> --out <path>"


def save_policy(path: str, net: PolicyNet, meta: dict | None = None) -> None:
    """
    Write `net` plus enough provenance to identify it a month later.

    A wrapper rather than a bare `state_dict` (which is what `DQN` saves) so
    that the two can never be mistaken for one another in either direction.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "format": POLICY_FORMAT,
            "input_size": net.INPUT_SIZE,
            "output_size": net.OUTPUT_SIZE,
            "state_dict": net.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_policy(
    path: str, device: torch.device, dropout: float = 0.1
) -> tuple[PolicyNet, dict[str, Any]]:
    """
    Restore a `PolicyNet`, refusing anything that is not one.

    Without this guard a DQN checkpoint - same 240 -> 32 shape, one flag away
    at `./trained_models/checkpoint_p0.pt` - loads cleanly and the agent plays
    Q-values as though they were policy logits. Wrong, and completely silent.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != POLICY_FORMAT:
        raise RuntimeError(
            f"{path} is not a policy checkpoint ({POLICY_FORMAT}).\n"
            "A bare state dict here is almost certainly a DQN checkpoint "
            "(./trained_models/checkpoint_p*.pt): same shape, different "
            "meaning - its outputs are Q-values, not policy logits.\n"
            f"Train a policy with:\n  {TRAIN_HINT}"
        )
    if blob["input_size"] != PolicyNet.INPUT_SIZE:
        raise RuntimeError(
            f"{path} was trained on {blob['input_size']}-wide observations, "
            f"`Round.get_state` is now {PolicyNet.INPUT_SIZE} wide. The "
            "checkpoint is retired - retrain."
        )
    net = PolicyNet(dropout=dropout).to(device)
    net.load_state_dict(blob["state_dict"])
    return net, blob.get("meta", {})
