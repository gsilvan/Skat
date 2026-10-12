"""Bounded versioned hand/outcome shards, independent of the game engine."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from skat.agents.distill.demonstrations import file_hash
from skat.agents.neural_bidding.features import FEATURE_SCHEMA, INPUT_SIZE
from skat.rules import RULES_SCHEMA

FORMAT = "skat-iss-bidding-corpus-v1"
OBJECTIVE = "iss-historical-solo-win-v1"
SPLITS = ("train", "val", "calibration", "test")


class BiddingWriter:
    def __init__(self, directory, shard_rows=16384):
        self.directory = Path(directory)
        self.shard_rows = shard_rows
        self.buffers = {s: [] for s in SPLITS}
        self.shards = []
        self.rows, self.wins = Counter(), Counter()

    def add(self, state, won, split, metadata):
        self.buffers[split].append((state, float(won)))
        self.rows[split] += 1
        self.wins[split] += int(won)
        with (self.directory / "games.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(split=split, **metadata)) + "\n")
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split):
        rows = self.buffers[split]
        if not rows:
            return
        name = f"{split}-{len(self.shards):05d}"
        folder = self.directory / name
        folder.mkdir()
        files = {}
        for column, key in enumerate(("states", "labels")):
            path = folder / f"{key}.npy"
            np.save(
                path,
                np.asarray([row[column] for row in rows], dtype=np.float32),
                allow_pickle=False,
            )
            files[path.name] = file_hash(path)
        self.shards.append(dict(path=name, split=split, rows=len(rows), sha256=files))
        self.buffers[split] = []

    def finish(self, provenance):
        if any(not self.rows[s] for s in SPLITS):
            raise ValueError(
                "bidding corpus requires nonempty train/val/calibration/test splits"
            )
        for split in SPLITS:
            self.flush(split)
        manifest = dict(
            format=FORMAT,
            feature_schema=FEATURE_SCHEMA,
            input_size=INPUT_SIZE,
            rules_schema=RULES_SCHEMA,
            objective=OBJECTIVE,
            shards=self.shards,
            rows=dict(self.rows),
            wins=dict(self.wins),
            provenance=provenance,
            games_sha256=file_hash(self.directory / "games.jsonl"),
        )
        path = self.directory / "manifest.json.tmp"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        path.replace(self.directory / "manifest.json")
        return manifest


class BiddingCorpus:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        m = self.manifest
        if (
            (
                m.get("format"),
                m.get("feature_schema"),
                m.get("input_size"),
                m.get("rules_schema"),
                m.get("objective"),
            )
            != (FORMAT, FEATURE_SCHEMA, INPUT_SIZE, RULES_SCHEMA, OBJECTIVE)
            or set(m["rows"]) != set(SPLITS)
            or any(m["rows"][s] < 1 for s in SPLITS)
        ):
            raise ValueError("incompatible or incomplete bidding corpus")
        if file_hash(self.directory / "games.jsonl") != m["games_sha256"]:
            raise ValueError("bidding metadata checksum mismatch")
        counts = Counter()
        seen = set()
        for shard in m["shards"]:
            if (
                shard["path"] in seen
                or Path(shard["path"]).name != shard["path"]
                or shard["path"] in (".", "..")
                or shard["split"] not in SPLITS
            ):
                raise ValueError("invalid bidding shard manifest")
            seen.add(shard["path"])
            data = self.open(shard, verify=True)
            counts[shard["split"]] += len(data["labels"])
        if dict(counts) != m["rows"]:
            raise ValueError("inconsistent bidding corpus counts")
        self.sha256 = file_hash(self.directory / "manifest.json")

    def open(self, shard, verify=False):
        if set(shard["sha256"]) != {"states.npy", "labels.npy"}:
            raise ValueError("invalid bidding shard columns")
        data = {}
        for key, shape in (
            ("states", (shard["rows"], INPUT_SIZE)),
            ("labels", (shard["rows"],)),
        ):
            path = self.directory / shard["path"] / f"{key}.npy"
            if verify and file_hash(path) != shard["sha256"][path.name]:
                raise ValueError("bidding shard checksum mismatch")
            data[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            if data[key].shape != shape or data[key].dtype != np.float32:
                raise ValueError("invalid bidding array layout")
        if verify:
            x, y = data["states"], data["labels"]
            if (
                not np.isin(x, (0, 1)).all()
                or not np.isin(y, (0, 1)).all()
                or not (x[:, :32].sum(1) == 10).all()
                or not (x[:, 32:].sum(1) == 1).all()
            ):
                raise ValueError("invalid original hand or outcome")
        return data

    def batches(self, split, batch_size, seed=None):
        rng = np.random.default_rng(seed)
        shards = [s for s in self.manifest["shards"] if s["split"] == split]
        if seed is not None:
            rng.shuffle(shards)
        for shard in shards:
            data = self.open(shard)
            indices = np.arange(shard["rows"])
            if seed is not None:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                rows = indices[start : start + batch_size]
                yield np.asarray(data["states"][rows]), np.asarray(data["labels"][rows])
