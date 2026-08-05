from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from skat.models import ARCHITECTURES, DEFAULT_ARCH

#: Written into every policy checkpoint. Its only job is to be absent from a
#: DQN checkpoint, which is a bare state dict.
#:
#: Deliberately not bumped when the second architecture arrived: the format did
#: not change, a key was added to it, and bumping would have retired the one
#: file holding the measured MLP baseline.
POLICY_FORMAT = "skat-policy-v1"

TRAIN_HINT = "python -m skat.agents.distill.train --data <dir> --out <path>"


def name_of(net: nn.Module) -> str:
    """The `ARCHITECTURES` key for `net`."""
    for name, architecture in ARCHITECTURES.items():
        if type(net) is architecture:
            return name
    raise ValueError(
        f"{type(net).__name__} is not in skat.models.ARCHITECTURES "
        f"({', '.join(sorted(ARCHITECTURES))}) - register it there first, or "
        "its checkpoints will not be loadable."
    )


def save_policy(
    path: str,
    net: nn.Module,
    meta: dict | None = None,
    arch: str | None = None,
    arch_kwargs: dict | None = None,
) -> None:
    """
    Write `net` plus enough provenance to identify it a month later.

    A wrapper rather than a bare `state_dict` (which is what `DQN` saves) so
    that the two can never be mistaken for one another in either direction.

    `arch` names the class in `skat.models.ARCHITECTURES` and `arch_kwargs`
    holds whatever `load_policy` has to pass back to rebuild it - the feature
    set, the depth. Not `dropout`: that one is the loader's to choose, since it
    is inert under `eval()` and every serving path evaluates.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "format": POLICY_FORMAT,
            "arch": arch or name_of(net),
            "arch_kwargs": dict(arch_kwargs or {}),
            "input_size": net.INPUT_SIZE,
            "output_size": net.OUTPUT_SIZE,
            "state_dict": net.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_policy(
    path: str, device: torch.device, dropout: float = 0.1
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Restore whichever policy network the file holds, refusing anything that is
    not a policy network at all.

    Without the format guard a DQN checkpoint - same 240 -> 32 shape, one flag
    away at `./trained_models/checkpoint_p0.pt` - loads cleanly and the agent
    plays Q-values as though they were policy logits. Wrong, and completely
    silent.

    Dispatching here rather than on a CLI flag is what keeps `--policy PATH`
    meaning "whatever is in this file", and is why `PolicyAgent` never had to
    learn that there is more than one architecture.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    # This check stays first: everything below assumes a policy blob.
    if not isinstance(blob, dict) or blob.get("format") != POLICY_FORMAT:
        raise RuntimeError(
            f"{path} is not a policy checkpoint ({POLICY_FORMAT}).\n"
            "A bare state dict here is almost certainly a DQN checkpoint "
            "(./trained_models/checkpoint_p*.pt): same shape, different "
            "meaning - its outputs are Q-values, not policy logits.\n"
            f"Train a policy with:\n  {TRAIN_HINT}"
        )
    # Absent from every checkpoint written before there was a second
    # architecture, which is what the default is for.
    arch = blob.get("arch", DEFAULT_ARCH)
    if arch not in ARCHITECTURES:
        raise RuntimeError(
            f"{path} was trained with architecture {arch!r}, which this build "
            f"does not have. Known: {', '.join(sorted(ARCHITECTURES))}.\n"
            "Either the checkpoint comes from a newer tree, or the name was "
            "renamed - it is written into every file, so renaming retires them."
        )
    architecture = ARCHITECTURES[arch]
    if blob["input_size"] != architecture.INPUT_SIZE:
        raise RuntimeError(
            f"{path} was trained on {blob['input_size']}-wide observations, "
            f"`Round.get_state` is now {architecture.INPUT_SIZE} wide. The "
            "checkpoint is retired - retrain."
        )
    net = architecture(**{**blob.get("arch_kwargs", {}), "dropout": dropout}).to(device)
    net.load_state_dict(blob["state_dict"])
    return net, blob.get("meta", {})
