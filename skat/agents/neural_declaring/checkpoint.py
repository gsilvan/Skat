"""Versioned declaration checkpoints, separate from bidding and card policies."""

import math
from pathlib import Path

import torch

from skat.agents.neural_declaring.features import FEATURE_SCHEMA, INPUT_SIZE
from skat.models.declaring import DeclarationNet
from skat.rules import RULES_SCHEMA

FORMAT = "skat-declaration-v1"
OBJECTIVE = "fixed-policy-counterfactual-solo-win-v1"


def save_declaration(path, model, *, temperature=1.0, metadata=None):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive calibration temperature required")
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
            temperature=temperature,
            state_dict=model.state_dict(),
            metadata=metadata or {},
        ),
        temporary,
    )
    temporary.replace(path)


def load_declaration(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(blob, dict) or (
        blob.get("format"),
        blob.get("feature_schema"),
        blob.get("input_size"),
        blob.get("rules_schema"),
        blob.get("objective"),
    ) != (FORMAT, FEATURE_SCHEMA, INPUT_SIZE, RULES_SCHEMA, OBJECTIVE):
        raise ValueError("incompatible declaration checkpoint")
    temperature = blob.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid declaration calibration")
    model = DeclarationNet(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model, float(temperature), blob.get("metadata", {})
