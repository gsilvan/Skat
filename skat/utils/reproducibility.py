"""Stable seeds and optional Torch setup."""

import hashlib
import os
import random

import numpy as np


def stage_seed(seed, stage):
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{stage}".encode()).digest()[:8], "big"
    ) % (2**32)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def resolve_device(device="cpu"):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA/ROCm device requested but unavailable")
    return result
