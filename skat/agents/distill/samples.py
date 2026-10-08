from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np
import torch

from skat.card import Card

#: Bumped whenever the array layout below changes in a way a reader cannot
#: absorb. `load_shards` refuses anything else.
SHARD_VERSION = 1

#: ~50k samples is roughly 20 MB compressed - small enough that a crash costs
#: minutes of collection, large enough that the file count stays sane.
DEFAULT_SHARD_SIZE = 50_000

#: Width of `Round.get_state`. Passed in rather than imported: this package
#: deals in arrays, not in rounds.
DEFAULT_INPUT_SIZE = 240

CARDS = 32


class Samples(NamedTuple):
    """Every shard in a directory, concatenated and moved onto tensors."""

    states: torch.Tensor  # (N, input_size) float32
    visits: torch.Tensor  # (N, 32) float32 - raw counts, not normalized
    values: torch.Tensor  # (N, 32) float32 - summed value, not averaged
    masks: torch.Tensor  # (N, 32) bool
    groups: torch.Tensor  # (N,) int64 - round id, for a leak-free split
    choices: torch.Tensor  # (N,) int64 - the card the teacher played
    reward_objective: str = "legacy-point-shaped"

    def __len__(self) -> int:
        return int(self.states.shape[0])


class ShardWriter:
    """
    Collects what `RecordingMCTSAgent`'s search believed and writes it out as
    compressed `.npz` shards.

    Deliberately array-shaped: it is handed a numpy observation and a plain
    `{Card: [visits, value_sum]}` tally, so it stays on the same side of the
    line as the rest of `skat.agents.distill` and knows nothing about `Round`,
    `Player` or `Agent`.

    Sharding buys two things. A collection run is an hour long, and a crash or
    a Ctrl-C in the middle of it must not cost the whole corpus; and several
    processes can collect into one directory at once, since the file names
    carry the pid.
    """

    def __init__(
        self,
        directory: str,
        shard_size: int = DEFAULT_SHARD_SIZE,
        input_size: int = DEFAULT_INPUT_SIZE,
        tag: str = "",
        provenance: dict[str, int | str] | None = None,
    ) -> None:
        self.directory = directory
        self.shard_size = shard_size
        self.input_size = input_size
        self.tag = str(tag)
        self.provenance = provenance or {}
        os.makedirs(directory, exist_ok=True)
        self.written = 0
        self.shards = 0
        self.group = -1
        self._states: list[np.ndarray] = []
        self._visits: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._groups: list[int] = []
        self._choices: list[int] = []

    def __len__(self) -> int:
        """Samples handed to `add` so far, buffered and flushed together."""
        return self.written + len(self._states)

    def new_round(self) -> None:
        """
        Start a new group.

        Every sample of one round shares a group id, so `train.py` can split
        train from validation on round boundaries. The ten decisions of a round
        share a hand, a soloist and a trick history; splitting them row-wise
        leaks the answer across the split and flatters the validation curve.

        All three agents fire this before a card is played, so the counter
        jumps by three per round. Only the boundaries matter.
        """
        self.group += 1

    def add(
        self,
        state: np.ndarray,
        totals: dict[Card, Sequence[float]],
        moves: Iterable[Card],
        choice: Card,
    ) -> None:
        """Record one searched decision. `totals` is `{move: [visits, value_sum]}`."""
        if len(state) != self.input_size:
            raise ValueError(
                f"observation is {len(state)} wide, this corpus is {self.input_size}"
            )
        visits = np.zeros(CARDS, dtype=np.int32)
        values = np.zeros(CARDS, dtype=np.float32)
        mask = np.zeros(CARDS, dtype=bool)
        for move in moves:
            visited, value_sum = totals[move]
            mask[move.np_index] = True
            visits[move.np_index] = visited
            values[move.np_index] = value_sum
        if not visits.sum():
            # A search that expanded nothing has no opinion to learn from. It
            # takes `iterations < 1` to get here, but a row of zeros would
            # divide to nan in `visit_targets`, so refuse it at the source.
            return
        self._states.append(np.asarray(state, dtype=np.float32))
        self._visits.append(visits)
        self._values.append(values)
        self._masks.append(mask)
        self._groups.append(self.group)
        self._choices.append(choice.np_index)
        if len(self._states) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Write the buffer out as one shard. A no-op when there is nothing to write."""
        if not self._states:
            return
        path = os.path.join(
            self.directory, f"mcts-{self.tag}-{os.getpid()}-{self.shards:04d}.npz"
        )
        arrays: dict[str, Any] = {
            "states": np.stack(self._states),
            "visits": np.stack(self._visits),
            "values": np.stack(self._values),
            "masks": np.stack(self._masks),
            "groups": np.asarray(self._groups, dtype=np.int32),
            "choices": np.asarray(self._choices, dtype=np.int8),
            "input_size": np.int32(self.input_size),
            "shard_version": np.int32(SHARD_VERSION),
        }
        # Which search settings produced this - the difference between a corpus
        # worth keeping and a mystery directory six months from now.
        # Seeds are derived from SHA-256 and intentionally span uint32.  A
        # signed int32 overflows for roughly half of that range (e.g. seed
        # 2522612516), aborting an otherwise completed collection run.
        arrays.update(
            {
                k: np.asarray(v) if isinstance(v, str) else np.int64(v)
                for k, v in self.provenance.items()
            }
        )
        np.savez_compressed(path, **arrays)
        self.written += len(self._states)
        self.shards += 1
        self._states.clear()
        self._visits.clear()
        self._values.clear()
        self._masks.clear()
        self._groups.clear()
        self._choices.clear()

    def close(self) -> None:
        self.flush()
        if not self.written:
            print(
                f"recorded nothing into {self.directory} - "
                "was there an `mcts` seat at the table?"
            )
            return
        print(
            f"recorded {self.written} decisions into {self.shards} shard(s) "
            f"under {self.directory}"
        )


def load_shards(
    directory: str, input_size: int = DEFAULT_INPUT_SIZE, pattern: str = ".npz"
) -> Samples:
    """
    Concatenate every shard in `directory`.

    ~100k samples is ~110 MB, so everything is loaded eagerly and batched by
    index permutation. A `DataLoader`'s per-item collate would dominate rows
    this small.
    """
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(pattern)
    )
    if not paths:
        raise FileNotFoundError(f"no shards under {directory}")

    states, visits, values, masks, groups, choices = [], [], [], [], [], []
    offset = 0
    objectives = set()
    for path in paths:
        with np.load(path) as shard:
            objectives.add(
                str(shard["reward_objective"])
                if "reward_objective" in shard
                else "legacy-point-shaped"
            )
            if len(objectives) > 1:
                raise RuntimeError(
                    "Mixed reward objectives in corpus; keep old and win-first shards in separate directories."
                )
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
            visits.append(shard["visits"])
            values.append(shard["values"])
            masks.append(shard["masks"])
            # Group ids restart at 0 in every shard and every process. Offset
            # them so a split on group boundaries stays a split.
            shard_groups = shard["groups"].astype(np.int64)
            groups.append(shard_groups + offset)
            offset = int(groups[-1].max()) + 1
            choices.append(shard["choices"])

    return Samples(
        states=torch.from_numpy(np.concatenate(states)).to(torch.float32),
        visits=torch.from_numpy(np.concatenate(visits)).to(torch.float32),
        values=torch.from_numpy(np.concatenate(values)).to(torch.float32),
        masks=torch.from_numpy(np.concatenate(masks)).to(torch.bool),
        groups=torch.from_numpy(np.concatenate(groups)).to(torch.int64),
        choices=torch.from_numpy(np.concatenate(choices)).to(torch.int64),
        reward_objective=next(iter(objectives)),
    )
