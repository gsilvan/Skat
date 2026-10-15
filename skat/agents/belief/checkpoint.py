"""Versioned belief checkpoints; never interchangeable with policy models."""

import os
from pathlib import Path

import torch

from skat.agents.belief.features import INPUT_SIZE, SCHEMA
from skat.agents.belief.model import BeliefTransformer
from skat.models.observation import OBSERVATION_SCHEMA
from skat.rules import RULES_SCHEMA

FORMAT = "skat-belief-iss-v1"


def save_belief(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        dict(
            format=FORMAT,
            input_schema=SCHEMA,
            input_size=INPUT_SIZE,
            observation_schema=OBSERVATION_SCHEMA,
            rules_schema=RULES_SCHEMA,
            config=model.config,
            state_dict=model.state_dict(),
            metadata=metadata,
        ),
        temporary,
    )
    os.replace(temporary, path)


def load_belief(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != FORMAT
        or checkpoint.get("input_schema") != SCHEMA
        or checkpoint.get("input_size") != INPUT_SIZE
        or checkpoint.get("observation_schema") != OBSERVATION_SCHEMA
        or checkpoint.get("rules_schema") != RULES_SCHEMA
    ):
        raise ValueError("incompatible belief checkpoint")
    model = BeliefTransformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint["metadata"]
