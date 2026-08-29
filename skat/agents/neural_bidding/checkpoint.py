from __future__ import annotations

import os
from typing import Any

import torch

from skat.models.bidding import BiddingNet

BIDDING_FORMAT = "skat-bidding-v1"
TRAIN_HINT = (
    "python -m skat.agents.neural_bidding.collect --out data/bidding && "
    "python -m skat.agents.neural_bidding.train --data data/bidding "
    "--out trained_models/bidding.pt"
)


def save_bidding(path: str, net: BiddingNet, meta: dict | None = None) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "format": BIDDING_FORMAT,
            "input_size": net.INPUT_SIZE,
            "hidden_size": net.hidden_size,
            "dropout": net.dropout,
            "state_dict": net.state_dict(),
            "meta": meta or {},
        },
        path,
    )


def load_bidding(path: str, device: torch.device) -> tuple[BiddingNet, dict[str, Any]]:
    blob = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(blob, dict) or blob.get("format") != BIDDING_FORMAT:
        raise RuntimeError(
            f"{path} is not a bidding checkpoint ({BIDDING_FORMAT}).\n"
            f"Train one with:\n  {TRAIN_HINT}"
        )
    if blob.get("input_size") != BiddingNet.INPUT_SIZE:
        raise RuntimeError(
            f"{path} uses {blob.get('input_size')} inputs; bidding now expects "
            f"{BiddingNet.INPUT_SIZE}. Recollect and retrain."
        )
    net = BiddingNet(
        hidden_size=int(blob.get("hidden_size", 128)),
        dropout=float(blob.get("dropout", 0.1)),
    ).to(device)
    net.load_state_dict(blob["state_dict"])
    return net, dict(blob.get("meta", {}))
