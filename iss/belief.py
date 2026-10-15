"""Stream hidden-hand labels from validated, selected ISS games."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from random import Random

import numpy as np

from iss.pipeline import records, select_games, sha256
from iss.protocol import ReplayRound, parse_record
from skat.agents.belief.features import INPUT_SIZE, SCHEMA, belief_input, locations
from skat.models.observation import OBSERVATION_SCHEMA
from skat.rules import RULES_SCHEMA
from skat.utils.reproducibility import stage_seed

FORMAT = "iss-belief-v1"


class BeliefWriter:
    def __init__(self, directory, shard_rows=8192):
        self.directory = Path(directory)
        self.shard_rows = shard_rows
        self.buffers = {split: [] for split in ("train", "val", "test")}
        self.shards = []
        self.rows = Counter()
        self.games = Counter()

    def add(self, split, examples):
        self.games[split] += 1
        for features, targets in examples:
            self.buffers[split].append((features, targets))
            self.rows[split] += 1
            if len(self.buffers[split]) >= self.shard_rows:
                self.flush(split)

    def flush(self, split):
        rows = self.buffers[split]
        if not rows:
            return
        folder = self.directory / f"{split}-{len(self.shards):05d}"
        folder.mkdir()
        paths = {}
        for name, array in (
            ("states", np.stack([row[0] for row in rows])),
            ("locations", np.stack([row[1] for row in rows])),
        ):
            path = folder / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            paths[path.name] = sha256(path)
        self.shards.append(
            dict(path=folder.name, split=split, rows=len(rows), sha256=paths)
        )
        self.buffers[split] = []

    def finish(self, provenance):
        for split in self.buffers:
            self.flush(split)
        if any(self.rows[split] == 0 for split in self.buffers):
            raise ValueError("belief corpus needs nonempty train/val/test partitions")
        manifest = dict(
            format=FORMAT,
            input_schema=SCHEMA,
            input_size=INPUT_SIZE,
            observation_schema=OBSERVATION_SCHEMA,
            rules_schema=RULES_SCHEMA,
            rows=dict(self.rows),
            games=dict(self.games),
            shards=self.shards,
            provenance=provenance,
        )
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(self.directory / "manifest.json")
        return manifest


def examples_from_record(record, seed=7):
    """Reservoir-select at most one non-forced turn in each third of a game."""
    round_ = ReplayRound(
        dealer=2,
        start=False,
        skip_bidding=True,
        solo_player_id=record.soloist,
        declare_game=record.contract,
        hand_game=True,
        replay=True,
        initial_cards=[*record.hands, record.skat],
    )
    round_.highest_bid = record.bid
    round_.prepare()
    rng = Random(stage_seed(seed, record.identifier))
    selected = [None, None, None]
    counts = [0, 0, 0]
    for index, (seat, card) in enumerate(record.actions):
        if round_.is_finished or round_.next_player != seat:
            raise ValueError("invalid ISS action order")
        phase = min(2, 3 * index // len(record.actions))
        if len(round_.player[seat].valid_moves(round_.game.trick)) > 1:
            counts[phase] += 1
            if rng.randrange(counts[phase]) == 0:
                view = round_.view(seat)
                selected[phase] = (
                    belief_input(view),
                    locations(view, [p.hand for p in round_.player], round_.skat),
                )
        round_.play_card(seat, card)
    if not round_.is_finished or round_.result is None:
        raise ValueError("incomplete ISS replay")
    return [row for row in selected if row is not None]


def import_belief(selection, archive, output, *, seed=7, progress=True):
    """Use an existing audited ISS selection; verify the archive's exact hash."""
    selection, archive, output = Path(selection), Path(archive), Path(output)
    source_manifest = json.loads((selection / "manifest.json").read_text())
    provenance = source_manifest["provenance"]
    if source_manifest.get("format") != "iss-action-v1":
        raise ValueError("expected an audited ISS action selection")
    if sha256(archive) != provenance["archive_sha256"]:
        raise ValueError("ISS archive differs from audited selection")
    if output.exists():
        raise ValueError("use a new belief corpus directory")
    output.mkdir(parents=True)
    return _import_selected(
        selection,
        archive,
        output,
        provenance,
        seed=seed,
        progress=progress,
        selection_manifest_sha256=sha256(selection / "manifest.json"),
    )


def import_belief_from_audit(
    audit, output, *, archive=None, max_games=300000, seed=7, progress=True
):
    """Select more audited games without materializing an action corpus."""
    audit, output = Path(audit), Path(output)
    if output.exists():
        raise ValueError("use a new belief corpus directory")
    output.mkdir(parents=True)
    selection = output / "selection"
    archive, provenance = select_games(
        audit,
        selection,
        max_games=max_games,
        seed=seed,
        archive_override=archive,
    )
    return _import_selected(
        selection,
        archive,
        output,
        provenance,
        seed=seed,
        progress=progress,
        audit_sha256=sha256(audit / "audit.json"),
        selection_index_sha256=sha256(selection / "index.sqlite"),
    )


def _import_selected(
    selection, archive, output, provenance, *, seed, progress, **source
):
    writer = BeliefWriter(output)
    db = sqlite3.connect(
        f"file:{(selection / 'index.sqlite').resolve()}?mode=ro", uri=True
    )
    try:
        targets = iter(db.execute("SELECT line, split FROM selection ORDER BY line"))
        target = next(targets, None)
        with records(archive, progress=progress) as lines:
            for index, line in lines:
                if target is None:
                    break
                if index != target[0]:
                    continue
                record = parse_record(line)
                writer.add(target[1], examples_from_record(record, seed))
                target = next(targets, None)
        if target is not None:
            raise ValueError("archive ended before selected ISS record")
    finally:
        db.close()
    return writer.finish(
        dict(
            source=provenance,
            archive_sha256=provenance["archive_sha256"],
            seed=seed,
            **source,
        )
    )


class BeliefCorpus:
    def __init__(self, directory):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        m = self.manifest
        if (
            m.get("format"),
            m.get("input_schema"),
            m.get("input_size"),
            m.get("observation_schema"),
            m.get("rules_schema"),
        ) != (FORMAT, SCHEMA, INPUT_SIZE, OBSERVATION_SCHEMA, RULES_SCHEMA):
            raise ValueError("incompatible belief corpus")
        actual = Counter()
        for shard in m["shards"]:
            self.open_shard(shard, verify=True)
            actual[shard["split"]] += shard["rows"]
        if dict(actual) != m["rows"]:
            raise ValueError("belief shard counts disagree with manifest")
        self.sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def open_shard(self, shard, verify=False):
        folder = self.directory / shard["path"]
        arrays = {}
        for name, dtype, shape in (
            ("states", np.float32, (shard["rows"], INPUT_SIZE)),
            ("locations", np.int8, (shard["rows"], 32)),
        ):
            path = folder / f"{name}.npy"
            if verify and sha256(path) != shard["sha256"][path.name]:
                raise ValueError("belief shard checksum mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("invalid belief shard shape or dtype")
            arrays[name] = array
        if verify and (
            not np.isfinite(arrays["states"]).all()
            or np.any(arrays["locations"] < -1)
            or np.any(arrays["locations"] > 2)
            or np.any((arrays["locations"] >= 0).sum(axis=1) == 0)
        ):
            raise ValueError("invalid belief observations or labels")
        return arrays

    def batches(self, split, batch_size, *, seed=None):
        rng = np.random.default_rng(seed)
        shards = [shard for shard in self.manifest["shards"] if shard["split"] == split]
        if seed is not None:
            rng.shuffle(shards)
        for shard in shards:
            arrays = self.open_shard(shard)
            indices = np.arange(shard["rows"])
            if seed is not None:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                rows = indices[start : start + batch_size]
                yield {name: np.asarray(array[rows]) for name, array in arrays.items()}
