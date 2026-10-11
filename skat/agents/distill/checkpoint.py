import os
from pathlib import Path

import torch

from skat.agents.distill.samples import REWARD_OBJECTIVE
from skat.models.card_transformer import CardTransformer
from skat.models.observation import (
    INPUT_SIZE,
    OBSERVATION_SCHEMA,
    validate_observation_schema,
)
from skat.rules import RULES_SCHEMA

FORMAT = "skat-transformer-policy-v3"


def save_policy(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        dict(
            format=FORMAT,
            observation_schema=OBSERVATION_SCHEMA,
            rules_schema=RULES_SCHEMA,
            input_size=INPUT_SIZE,
            objective=REWARD_OBJECTIVE,
            config=model.config,
            state_dict=model.state_dict(),
            metadata=metadata or {},
        ),
        temporary,
    )
    os.replace(temporary, path)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError("incompatible policy format; retrain from a current corpus")
    validate_observation_schema(checkpoint.get("observation_schema"), path)
    if (
        checkpoint.get("rules_schema") != RULES_SCHEMA
        or checkpoint.get("input_size") != INPUT_SIZE
        or checkpoint.get("objective") != REWARD_OBJECTIVE
    ):
        raise ValueError("incompatible policy rules/layout/objective")
    model = CardTransformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint.get("metadata", {})
