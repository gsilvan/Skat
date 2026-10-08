from __future__ import annotations

import os
from typing import Any

import torch

from skat.models.belief import BeliefNet
from skat.models.observation import OBSERVATION_SCHEMA, validate_observation_schema

#: Distinct from `skat-policy-v1` on purpose. A `BeliefNet` is a subclass of
#: `CardTransformer` and its checkpoint is the same shape everywhere except the
#: head, so without a separate marker the two would be one careless `--policy`
#: away from loading into each other.
BELIEF_FORMAT = "skat-belief-v1"

TRAIN_HINT = "python -m skat.agents.belief.train --data <dir> --out <path>"


def save_belief(path: str, net: BeliefNet, meta: dict | None = None) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "format": BELIEF_FORMAT,
            "observation_schema": OBSERVATION_SCHEMA,
            "input_size": net.INPUT_SIZE,
            "locations": net.LOCATIONS,
            "state_dict": net.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_belief(
    path: str, device: torch.device, dropout: float = 0.1
) -> tuple[BeliefNet, dict[str, Any]]:
    """Restore a `BeliefNet`, refusing anything that is not one."""
    blob = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != BELIEF_FORMAT:
        raise RuntimeError(
            f"{path} is not a belief checkpoint ({BELIEF_FORMAT}).\n"
            "A policy checkpoint (skat-policy-v1) is nearly the same network - "
            "same tokeniser, same encoder - but its head answers 'which card "
            "should I play', not 'where is this card'.\n"
            f"Train one with:\n  {TRAIN_HINT}"
        )
    if blob["input_size"] != BeliefNet.INPUT_SIZE:
        raise RuntimeError(
            f"{path} was trained on {blob['input_size']}-wide observations, "
            f"`Round.get_state` is now {BeliefNet.INPUT_SIZE} wide. The "
            "checkpoint is retired - retrain."
        )
    validate_observation_schema(blob.get("observation_schema"), path, allow_legacy=True)
    net = BeliefNet(dropout=dropout).to(device)
    net.load_state_dict(blob["state_dict"])
    return net, blob.get("meta", {})
