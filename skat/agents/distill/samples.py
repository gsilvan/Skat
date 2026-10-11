"""Versioned atomic shards; group ids preserve complete-round splits."""

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skat.models.observation import (
    INPUT_SIZE,
    OBSERVATION_SCHEMA,
    validate_observation_schema,
)
from skat.rules import RULES_SCHEMA

SHARD_VERSION = 3
REWARD_OBJECTIVE = "party-victory-v1"


@dataclass
class Samples:
    states: np.ndarray
    visits: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    groups: np.ndarray
    choices: np.ndarray

    def __len__(self):
        return len(self.states)


class ShardWriter:
    def __init__(self, directory, shard_size=5000, provenance=None):
        if type(shard_size) is not int or shard_size < 1:
            raise ValueError("positive integer shard_size required")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.provenance = provenance or {}
        self.tag = uuid.uuid4().hex
        self.group = -1
        self.rows = []
        self.written = self.shards = 0
        # Starting a new collection in a used directory mixes deals/seeds and
        # breaks comparisons. Parallel writers are intentionally a later feature.
        if list(self.directory.glob("*.npz")):
            raise ValueError("use a new corpus directory for each collection")

    def __len__(self):
        return self.written + len(self.rows)

    def new_round(self):
        self.group += 1

    def add(self, state, totals, moves, choice):
        if self.group < 0:
            raise ValueError("start a round before recording")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (INPUT_SIZE,) or not np.isfinite(state).all():
            raise ValueError("invalid observation")
        visits = np.zeros(32, dtype=np.int32)
        values = np.zeros(32, dtype=np.float32)
        mask = np.zeros(32, dtype=bool)
        for move in moves:
            n, value = totals[move]
            if n < 0 or int(n) != n or not 0 <= value <= n:
                raise ValueError("invalid binary-outcome search tally")
            i = move.np_index
            visits[i], values[i], mask[i] = int(n), value, True
        if not visits.sum():
            return
        if not mask[choice.np_index]:
            raise ValueError("teacher choice must be legal")
        self.rows.append(
            (state.copy(), visits, values, mask, self.group, choice.np_index)
        )
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        columns = list(zip(*self.rows))
        target = self.directory / f"mcts-{self.tag}-{self.shards:05}.npz"
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                version=SHARD_VERSION,
                input_size=INPUT_SIZE,
                observation_schema=OBSERVATION_SCHEMA,
                rules_schema=RULES_SCHEMA,
                reward_objective=REWARD_OBJECTIVE,
                collector=self.tag,
                provenance=json.dumps(self.provenance, sort_keys=True),
                states=np.asarray(columns[0], dtype=np.float32),
                visits=np.asarray(columns[1]),
                values=np.asarray(columns[2]),
                masks=np.asarray(columns[3]),
                groups=np.asarray(columns[4], dtype=np.int64),
                choices=np.asarray(columns[5], dtype=np.int64),
            )
        os.replace(temporary, target)
        self.written += len(self.rows)
        self.shards += 1
        self.rows.clear()

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _shard_order(path):
    with np.load(path, allow_pickle=False) as shard:
        provenance = (
            json.loads(shard["provenance"].item()) if "provenance" in shard else {}
        )
    if not isinstance(provenance, dict):
        raise ValueError(f"{path}: invalid provenance")
    if "round_start" in provenance:
        start = provenance["round_start"]
        if type(start) is not int or start < 0:
            raise ValueError(f"{path}: invalid collection round offset")
        # Independent workers have random collector tags. Global round order
        # keeps row order and the seeded train/validation split reproducible.
        return (0, start, path.name)
    return (1, 0, path.name)


def load_shards(directory):
    files = sorted(Path(directory).glob("*.npz"), key=_shard_order)
    if not files:
        raise ValueError("no corpus shards found")
    columns = {
        name: []
        for name in ("states", "visits", "values", "masks", "groups", "choices")
    }
    group_map = {}
    for path in files:
        with np.load(path, allow_pickle=False) as shard:
            required = set(columns) | {
                "version",
                "input_size",
                "observation_schema",
                "rules_schema",
                "reward_objective",
                "collector",
            }
            if not required <= set(shard.files):
                raise ValueError(f"{path}: missing versioned corpus metadata")
            if (
                shard["version"].item() != SHARD_VERSION
                or shard["input_size"].item() != INPUT_SIZE
            ):
                raise ValueError(f"{path}: incompatible corpus format")
            validate_observation_schema(shard["observation_schema"].item(), path)
            if (
                shard["rules_schema"].item() != RULES_SCHEMA
                or shard["reward_objective"].item() != REWARD_OBJECTIVE
            ):
                raise ValueError(f"{path}: incompatible rules or training objective")
            n = len(shard["states"])
            expected = dict(
                states=(n, INPUT_SIZE),
                visits=(n, 32),
                values=(n, 32),
                masks=(n, 32),
                groups=(n,),
                choices=(n,),
            )
            if not n or any(shard[k].shape != shape for k, shape in expected.items()):
                raise ValueError(f"{path}: invalid array shapes")
            states, visits, values, masks = (
                shard[k] for k in ("states", "visits", "values", "masks")
            )
            choices = shard["choices"]
            if (
                masks.dtype != np.bool_
                or not np.issubdtype(shard["groups"].dtype, np.integer)
                or (shard["groups"] < 0).any()
                or not np.isfinite(states).all()
                or not np.isfinite(values).all()
                or not np.isfinite(visits).all()
                or (visits < 0).any()
                or not np.equal(visits, np.floor(visits)).all()
                or (values < 0).any()
                or (values > visits).any()
                or (visits[~masks] != 0).any()
                or (values[~masks] != 0).any()
                or (visits.sum(1) == 0).any()
                or not np.issubdtype(choices.dtype, np.integer)
                or ((choices < 0) | (choices >= 32)).any()
                or not masks[np.arange(n), choices].all()
            ):
                raise ValueError(f"{path}: invalid states, search tallies or choices")
            collector = str(shard["collector"].item())
            # Preserve a round spanning shard boundaries. Different collectors
            # may reuse the same local id without merging their rounds.
            groups = np.asarray(
                [
                    group_map.setdefault((collector, int(g)), len(group_map))
                    for g in shard["groups"]
                ],
                dtype=np.int64,
            )
            for name in columns:
                columns[name].append(groups if name == "groups" else shard[name].copy())
    return Samples(**{name: np.concatenate(values) for name, values in columns.items()})
