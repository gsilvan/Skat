"""Small typed configuration; common defaults live here, not in argparse."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 600
    determinizations: int = 20

    def __post_init__(self):
        if (
            type(self.simulations) is not int
            or type(self.determinizations) is not int
            or not 1 <= self.determinizations <= self.simulations
        ):
            raise ValueError(
                "MCTS requires integer simulations >= determinizations >= 1"
            )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 0.0003
    dropout: float = 0.1
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    threads: int = 1

    def __post_init__(self):
        for name in (
            "epochs",
            "batch_size",
            "d_model",
            "nhead",
            "num_layers",
            "threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"train.{name} must be a positive integer")
        if (
            self.d_model % self.nhead
            or type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or type(self.dropout) not in (int, float)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("invalid training dimensions, learning rate or dropout")


def load_config(path=None):
    data = tomllib.loads(Path(path).read_text()) if path else {}
    if set(data) - {"mcts", "train"}:
        raise ValueError("configuration only supports [mcts] and [train]")
    configs = []
    for name, cls in (("mcts", MCTSConfig), ("train", TrainConfig)):
        values = data.get(name, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown options in [{name}]")
        configs.append(cls(**values))
    return tuple(configs)
