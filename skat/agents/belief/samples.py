from __future__ import annotations

import os
from typing import Any, NamedTuple

import numpy as np
import torch

#: Bumped when the array layout changes in a way a reader cannot absorb.
SHARD_VERSION = 1

DEFAULT_SHARD_SIZE = 200_000

#: `Round.get_state` (240) plus the two opponents' 32-wide void masks, which
#: the observation cannot express - see `skat.models.belief.BeliefNet`. Passed
#: in rather than imported: like `skat.agents.distill`, this package deals in
#: arrays, not in rounds.
DEFAULT_INPUT_SIZE = 240 + 2 * 32

CARDS = 32

#: A card the observer can already see - in its own hand, already played, or
#: in a skat it pressed itself. It has no location to predict, and the loss
#: skips it.
KNOWN = -1


class BeliefSamples(NamedTuple):
    """Every shard in a directory, concatenated onto tensors."""

    states: torch.Tensor  # (N, input_size) float32
    locations: torch.Tensor  # (N, 32) int64 - NEXT/PREVIOUS/SKAT, or KNOWN
    groups: torch.Tensor  # (N,) int64 - round id, for a leak-free split

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def unseen(self) -> torch.Tensor:
        """(N, 32) bool - the cards there is anything to predict about."""
        return self.locations >= 0


class BeliefShardWriter:
    """
    Collects `(observation, where every unseen card actually was)` and writes
    compressed `.npz` shards.

    Deliberately a separate writer from `skat.agents.distill.samples`
    rather than a generalisation of it: the two share their file-naming and
    guard conventions but almost nothing else. That one is handed a
    `{Card: tally}` mapping and scatters it over 32 slots; this one is handed a
    finished label array. Forcing them into one class would mean a writer that
    describes neither.

    One row per decision, and unlike the policy corpus that includes forced
    moves - a position is just as informative about where the cards are whether
    or not its owner had a choice.
    """

    def __init__(
        self,
        directory: str,
        shard_size: int = DEFAULT_SHARD_SIZE,
        input_size: int = DEFAULT_INPUT_SIZE,
        tag: str = "",
    ) -> None:
        self.directory = directory
        self.shard_size = shard_size
        self.input_size = input_size
        self.tag = str(tag)
        os.makedirs(directory, exist_ok=True)
        self.written = 0
        self.shards = 0
        self.group = -1
        self._states: list[np.ndarray] = []
        self._locations: list[np.ndarray] = []
        self._groups: list[int] = []

    def __len__(self) -> int:
        return self.written + len(self._states)

    def new_round(self) -> None:
        """Start a new group, so the split can cut on round boundaries."""
        self.group += 1

    def add(self, state: np.ndarray, locations: np.ndarray) -> None:
        """One decision: the observation, and where each card really was."""
        if len(state) != self.input_size:
            raise ValueError(
                f"observation is {len(state)} wide, this corpus is {self.input_size}"
            )
        if len(locations) != CARDS:
            raise ValueError(f"expected {CARDS} locations, got {len(locations)}")
        if not (locations >= 0).any():
            # Nothing hidden - the last trick of a round, where every card is
            # accounted for. There is no question to answer.
            return
        self._states.append(np.asarray(state, dtype=np.float32))
        self._locations.append(np.asarray(locations, dtype=np.int8))
        self._groups.append(self.group)
        if len(self._states) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._states:
            return
        path = os.path.join(
            self.directory, f"belief-{self.tag}-{os.getpid()}-{self.shards:04d}.npz"
        )
        arrays: dict[str, Any] = {
            "states": np.stack(self._states),
            "locations": np.stack(self._locations),
            "groups": np.asarray(self._groups, dtype=np.int32),
            "input_size": np.int32(self.input_size),
            "shard_version": np.int32(SHARD_VERSION),
        }
        np.savez_compressed(path, **arrays)
        self.written += len(self._states)
        self.shards += 1
        self._states.clear()
        self._locations.clear()
        self._groups.clear()

    def close(self) -> None:
        self.flush()
        if not self.written:
            print(f"recorded no beliefs into {self.directory}")
            return
        print(
            f"recorded {self.written} observations into {self.shards} shard(s) "
            f"under {self.directory}"
        )


def load_belief_shards(
    directory: str, input_size: int = DEFAULT_INPUT_SIZE
) -> BeliefSamples:
    """Concatenate every belief shard in `directory`."""
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".npz")
    )
    if not paths:
        raise FileNotFoundError(f"no belief shards under {directory}")

    states, locations, groups = [], [], []
    offset = 0
    for path in paths:
        with np.load(path) as shard:
            version = int(shard["shard_version"])
            if version != SHARD_VERSION:
                raise RuntimeError(
                    f"{path} is a v{version} shard, this reader speaks "
                    f"v{SHARD_VERSION}."
                )
            width = int(shard["input_size"])
            if width != input_size:
                raise RuntimeError(
                    f"{path} holds {width}-wide observations, the network "
                    f"expects {input_size}. `Round.get_state` changed shape "
                    "since this was collected, which retires it - recollect."
                )
            states.append(shard["states"])
            locations.append(shard["locations"])
            # Group ids restart at 0 in every shard and every process.
            shard_groups = shard["groups"].astype(np.int64)
            groups.append(shard_groups + offset)
            offset = int(groups[-1].max()) + 1

    return BeliefSamples(
        states=torch.from_numpy(np.concatenate(states)).to(torch.float32),
        locations=torch.from_numpy(np.concatenate(locations)).to(torch.int64),
        groups=torch.from_numpy(np.concatenate(groups)).to(torch.int64),
    )
