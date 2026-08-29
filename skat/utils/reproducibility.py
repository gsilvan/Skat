"""Utilities for repeatable training and evaluation runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int | None, deterministic: bool = True) -> None:
    """Seed every global RNG used by the project.

    Agents should still own private generators.  The global seed covers model
    initialisation and third-party code, while private generators keep one
    agent's random choices from perturbing another one's stream.

    Deterministic kernels make repeated runs on the *same* software and device
    reproducible.  CPU, CUDA and ROCm are not promised to be bit-identical to
    each other, so benchmark manifests record the selected device and versions.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUDA needs this before the first cuBLAS workspace is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
