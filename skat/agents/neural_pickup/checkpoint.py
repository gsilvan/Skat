"""Versioned counterfactual pickup checkpoints."""

import math
from pathlib import Path

import torch

from skat.agents.neural_pickup.features import FEATURE_SCHEMA, INPUT_SIZE
from skat.models.pickup import PickupNet
from skat.rules import RULES_SCHEMA

FORMAT = "skat-pickup-choice-v1"
OBJECTIVE = "paired-fixed-policy-solo-win-v1"


def save_pickup(path, model, *, temperatures=(1.0, 1.0), metadata=None):
    if len(temperatures) != 2 or any(
        not math.isfinite(value) or value <= 0 for value in temperatures
    ):
        raise ValueError("two positive calibration temperatures required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        dict(
            format=FORMAT,
            feature_schema=FEATURE_SCHEMA,
            input_size=INPUT_SIZE,
            rules_schema=RULES_SCHEMA,
            objective=OBJECTIVE,
            config=model.config,
            temperatures=tuple(temperatures),
            state_dict=model.state_dict(),
            metadata=metadata or {},
        ),
        temporary,
    )
    temporary.replace(path)


def load_pickup(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(blob, dict) or (
        blob.get("format"),
        blob.get("feature_schema"),
        blob.get("input_size"),
        blob.get("rules_schema"),
        blob.get("objective"),
    ) != (FORMAT, FEATURE_SCHEMA, INPUT_SIZE, RULES_SCHEMA, OBJECTIVE):
        raise ValueError("incompatible pickup checkpoint")
    temperatures = blob.get("temperatures")
    if (
        not isinstance(temperatures, (tuple, list))
        or len(temperatures) != 2
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in temperatures
        )
    ):
        raise ValueError("invalid pickup calibration")
    model = PickupNet(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model, tuple(float(v) for v in temperatures), blob.get("metadata", {})
