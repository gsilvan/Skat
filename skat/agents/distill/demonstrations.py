"""Bounded, memory-mappable action demonstrations; no engine/agent imports."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from skat.models.observation import INPUT_SIZE, OBSERVATION_SCHEMA
from skat.rules import RULES_SCHEMA

FORMAT = "iss-action-v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DemonstrationWriter:
    def __init__(self, directory, shard_rows=16384):
        self.directory = Path(directory)
        self.shard_rows = shard_rows
        self.buffers = {s: [] for s in ("train", "val", "test")}
        self.shards = []
        self.counts = Counter()
        self.uniform = Counter()
        self.coverage = {s: Counter() for s in self.buffers}
        self.games = self.directory / "games.jsonl"

    def add_game(self, group, split, metadata, rows):
        with self.games.open("a") as stream:
            stream.write(json.dumps(dict(group=group, split=split, **metadata)) + "\n")
        kind = {"suit": 0, "grand": 1, "null": 2}[metadata["declaration"]["kind"]]
        for state, mask, choice, seat in rows:
            solo = seat == metadata["soloist"]
            self.buffers[split].append((state, mask, choice, group, seat, solo, kind))
            self.counts[split] += 1
            self.uniform[split] += float(np.log(mask.sum()))
            self.coverage[split][
                f"{'soloist' if solo else 'defence'}/{metadata['declaration']['kind']}"
            ] += 1
            if len(self.buffers[split]) >= self.shard_rows:
                self.flush(split)

    def flush(self, split):
        rows = self.buffers[split]
        if not rows:
            return
        name = f"{split}-{len(self.shards):05d}"
        folder = self.directory / name
        folder.mkdir()
        specs = (
            ("states", np.float32),
            ("masks", bool),
            ("choices", np.int64),
            ("groups", np.int64),
            ("seats", np.uint8),
            ("solo", bool),
            ("kinds", np.uint8),
        )
        hashes = {}
        for column, (key, dtype) in enumerate(specs):
            path = folder / f"{key}.npy"
            np.save(
                path,
                np.asarray([row[column] for row in rows], dtype=dtype),
                allow_pickle=False,
            )
            hashes[path.name] = file_hash(path)
        self.shards.append(dict(path=name, split=split, rows=len(rows), sha256=hashes))
        self.buffers[split] = []

    def finish(self, provenance):
        for split in self.buffers:
            self.flush(split)
        manifest = dict(
            format=FORMAT,
            observation_schema=OBSERVATION_SCHEMA,
            rules_schema=RULES_SCHEMA,
            input_size=INPUT_SIZE,
            target="recorded-legal-action",
            shards=self.shards,
            rows=dict(self.counts),
            coverage={k: dict(v) for k, v in self.coverage.items()},
            target_baselines={
                s: dict(
                    target_entropy=0.0, uniform_loss=self.uniform[s] / self.counts[s]
                )
                for s in self.buffers
                if self.counts[s]
            },
            games_sha256=file_hash(self.games),
            provenance=provenance,
        )
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(self.directory / "manifest.json")
        return manifest


class Demonstrations:
    """At most one mapped shard plus one copied batch lives in memory."""

    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        if (
            m.get("format"),
            m.get("observation_schema"),
            m.get("rules_schema"),
            m.get("input_size"),
            m.get("target"),
        ) != (
            FORMAT,
            OBSERVATION_SCHEMA,
            RULES_SCHEMA,
            INPUT_SIZE,
            "recorded-legal-action",
        ):
            raise ValueError("incompatible ISS corpus format/rules/observation/target")
        if set(m["rows"]) != {"train", "val", "test"} or any(
            n < 1 for n in m["rows"].values()
        ):
            raise ValueError("ISS corpus requires nonempty train/val/test partitions")
        if file_hash(self.directory / "games.jsonl") != m["games_sha256"]:
            raise ValueError("ISS game metadata checksum mismatch")
        counts = Counter()
        seen = set()
        for shard in m["shards"]:
            if (
                shard["path"] in seen
                or Path(shard["path"]).name != shard["path"]
                or shard["split"] not in m["rows"]
            ):
                raise ValueError("invalid ISS shard manifest")
            seen.add(shard["path"])
            arrays = self.open_shard(shard, verify=True)
            counts[shard["split"]] += len(arrays["choices"])
        if dict(counts) != m["rows"]:
            raise ValueError("ISS row counts disagree with manifest")
        digest = hashlib.sha256(path.read_bytes())
        self.sha256 = digest.hexdigest()

    def open_shard(self, shard, *, verify=False):
        specs = dict(
            states=(np.float32, (shard["rows"], INPUT_SIZE)),
            masks=(bool, (shard["rows"], 32)),
            choices=(np.int64, (shard["rows"],)),
            groups=(np.int64, (shard["rows"],)),
            seats=(np.uint8, (shard["rows"],)),
            solo=(bool, (shard["rows"],)),
            kinds=(np.uint8, (shard["rows"],)),
        )
        if set(shard["sha256"]) != {f"{name}.npy" for name in specs}:
            raise ValueError("invalid ISS shard columns")
        arrays = {}
        for name, (dtype, shape) in specs.items():
            path = self.directory / shard["path"] / f"{name}.npy"
            if verify and file_hash(path) != shard["sha256"][path.name]:
                raise ValueError("ISS shard checksum mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.dtype(dtype) or array.shape != shape:
                raise ValueError("invalid ISS array layout")
            arrays[name] = array
        if verify:
            choices, masks = arrays["choices"], arrays["masks"]
            if (
                np.any(choices < 0)
                or np.any(choices >= 32)
                or np.any(masks.sum(1) < 2)
                or not masks[np.arange(len(choices)), choices].all()
                or not np.isfinite(arrays["states"]).all()
                or np.any(arrays["seats"] > 2)
                or np.any(arrays["kinds"] > 2)
            ):
                raise ValueError("invalid ISS actions/observations")
        return arrays

    def batches(self, split, batch_size, *, seed=None):
        # Shuffle shard order and rows inside a shard, preserving bounded RAM.
        rng = np.random.default_rng(seed)
        shards = [s for s in self.manifest["shards"] if s["split"] == split]
        if seed is not None:
            rng.shuffle(shards)
        for shard in shards:
            arrays = self.open_shard(shard)
            indices = np.arange(shard["rows"])
            if seed is not None:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                yield {
                    name: np.asarray(value[selected]) for name, value in arrays.items()
                }
