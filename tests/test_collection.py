import json

import numpy as np
import pytest

from skat.collection import collect
from skat.config import MCTSConfig


def corpus_rows(directory):
    columns = {
        key: [] for key in ("states", "visits", "values", "masks", "groups", "choices")
    }
    for file in sorted(directory.glob("*.npz")):
        with np.load(file, allow_pickle=False) as data:
            for key in columns:
                columns[key].append(data[key])
    columns = {key: np.concatenate(values) for key, values in columns.items()}
    order = np.argsort(columns["groups"], kind="stable")
    return {key: values[order] for key, values in columns.items()}


def test_parallel_collection_matches_serial_searches_and_round_groups(tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    config = MCTSConfig(12, 2)
    a = collect(serial, rounds=4, seed=9, mcts=config, workers=1, progress=False)
    b = collect(parallel, rounds=4, seed=9, mcts=config, workers=2, progress=False)
    assert a["completed_rounds"] == b["completed_rounds"] == 4
    assert a["decisions"] == b["decisions"] > 0
    assert b["status"] == "complete" and b["workers"] == 2
    first, second = corpus_rows(serial), corpus_rows(parallel)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    from skat.agents.distill.samples import load_shards

    loaded_serial, loaded_parallel = load_shards(serial), load_shards(parallel)
    for key in first:
        np.testing.assert_array_equal(
            getattr(loaded_serial, key), getattr(loaded_parallel, key)
        )
    assert not list(parallel.glob(".worker-*"))


def test_automatic_workers_are_limited_to_round_count(tmp_path, monkeypatch):
    monkeypatch.setattr("skat.collection.available_cpus", lambda: 16)
    result = collect(tmp_path / "data", rounds=2, mcts=MCTSConfig(4, 1), progress=False)
    assert result["workers"] == result["completed_rounds"] == 2


def test_collection_failure_flushes_completed_decisions(tmp_path, monkeypatch):
    from skat.game import Round

    calls = 0

    def fail_second_round(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("round failed")
        return Round(**kwargs)

    monkeypatch.setattr("skat.collection.Round", fail_second_round)
    directory = tmp_path / "data"
    with pytest.raises(RuntimeError, match="round failed"):
        collect(
            directory,
            rounds=3,
            seed=9,
            mcts=MCTSConfig(12, 2),
            workers=1,
            progress=False,
        )
    metadata = json.loads((directory / "collection.json").read_text())
    assert metadata["status"] == "failed" and metadata["completed_rounds"] == 1
    assert metadata["decisions"] > 0
    assert len(corpus_rows(directory)["states"]) == metadata["decisions"]
    assert not list(directory.glob(".worker-*"))


def test_collection_preserves_existing_runs_and_validates_workers(tmp_path):
    directory = tmp_path / "data"
    collect(directory, rounds=0, workers=1, progress=False)
    with pytest.raises(FileExistsError):
        collect(directory, rounds=0, workers=1, progress=False)
    for workers in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            collect(tmp_path / "invalid", workers=workers, progress=False)
    assert not (tmp_path / "invalid").exists()
