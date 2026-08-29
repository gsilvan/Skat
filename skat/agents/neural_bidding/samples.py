from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import torch

from skat.models.bidding import BiddingNet

SHARD_VERSION = 1
DEFAULT_SHARD_SIZE = 10_000


class BiddingSamples(NamedTuple):
    states: torch.Tensor
    labels: torch.Tensor
    rewards: torch.Tensor
    groups: torch.Tensor

    def __len__(self) -> int:
        return int(self.states.shape[0])


class BiddingShardWriter:
    def __init__(
        self,
        directory: str,
        shard_size: int = DEFAULT_SHARD_SIZE,
        tag: str = "",
        provenance: dict[str, int] | None = None,
    ) -> None:
        self.directory = directory
        self.shard_size = shard_size
        self.tag = str(tag)
        self.provenance = provenance or {}
        os.makedirs(directory, exist_ok=True)
        self.written = 0
        self.shards = 0
        self._states: list[np.ndarray] = []
        self._rewards: list[int] = []
        self._groups: list[int] = []

    def add(self, state: np.ndarray, won: bool, group: int) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (BiddingNet.INPUT_SIZE,):
            raise ValueError(
                f"bidding observation is {state.shape}, expected "
                f"({BiddingNet.INPUT_SIZE},)"
            )
        # Never split one deal over two shards: the loader offsets shard-local
        # group ids, so splitting here would make siblings eligible for both
        # training and validation. A shard may exceed its target by at most one
        # deal (three rows).
        if (
            self._states
            and len(self._states) >= self.shard_size
            and group != self._groups[-1]
        ):
            self.flush()
        self._states.append(state)
        self._rewards.append(1 if won else -1)
        self._groups.append(group)

    def flush(self) -> None:
        if not self._states:
            return
        path = os.path.join(
            self.directory, f"bidding-{self.tag}-{os.getpid()}-{self.shards:04d}.npz"
        )
        arrays = {
            "states": np.stack(self._states),
            "rewards": np.asarray(self._rewards, dtype=np.int8),
            "groups": np.asarray(self._groups, dtype=np.int32),
            "input_size": np.int32(BiddingNet.INPUT_SIZE),
            "shard_version": np.int32(SHARD_VERSION),
            # Stage seeds are full uint32 values; int64 keeps them lossless on
            # NumPy versions that reject out-of-range int32 construction.
            **{key: np.int64(value) for key, value in self.provenance.items()},
        }
        np.savez_compressed(path, **arrays)
        self.written += len(self._states)
        self.shards += 1
        self._states.clear()
        self._rewards.clear()
        self._groups.clear()

    def close(self) -> None:
        self.flush()


def load_bidding_shards(directory: str) -> BiddingSamples:
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".npz")
    )
    if not paths:
        raise FileNotFoundError(f"no bidding shards under {directory}")
    states, rewards, groups = [], [], []
    offset = 0
    for path in paths:
        with np.load(path) as shard:
            if int(shard["shard_version"]) != SHARD_VERSION:
                raise RuntimeError(f"unsupported bidding shard version in {path}")
            if int(shard["input_size"]) != BiddingNet.INPUT_SIZE:
                raise RuntimeError(
                    f"{path} has {int(shard['input_size'])} inputs; expected "
                    f"{BiddingNet.INPUT_SIZE}"
                )
            states.append(shard["states"])
            rewards.append(shard["rewards"])
            shard_groups = shard["groups"].astype(np.int64)
            groups.append(shard_groups + offset)
            offset = int(groups[-1].max()) + 1
    reward_tensor = torch.from_numpy(np.concatenate(rewards)).to(torch.float32)
    return BiddingSamples(
        states=torch.from_numpy(np.concatenate(states)).to(torch.float32),
        labels=(reward_tensor > 0).to(torch.float32),
        rewards=reward_tensor,
        groups=torch.from_numpy(np.concatenate(groups)).to(torch.int64),
    )
