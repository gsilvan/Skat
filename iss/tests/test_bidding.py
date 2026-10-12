import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pysgf")

from iss.bidding import import_bidding
from iss.pipeline import audit
from iss.protocol import parse_record
from skat.agents.neural_bidding.features import bidding_state
from skat.agents.neural_bidding.samples import BiddingCorpus

FIXTURE = Path(__file__).parent / "fixtures/bidding-games.sgf"


def test_import_keeps_original_ten_cards_solo_labels_and_whole_groups(tmp_path):
    lines = []
    for i, line in enumerate(FIXTURE.read_text().splitlines(keepends=True)):
        date = (datetime(2020, 1, 1) + timedelta(days=i)).strftime(
            "%Y-%m-%d/%H:%M:%S/UTC"
        )
        line = re.sub(r"ID\[[^]]*\]", f"ID[{i}]", line)
        line = re.sub(r"SE\[[^]]*\]", f"SE[session-{i}]", line)
        line = re.sub(r"DT\[[^]]*\]", f"DT[{date}]", line)
        lines.append(line)
    archive = tmp_path / "input.sgf"
    archive.write_text("".join(lines))
    audited = tmp_path / "audit"
    audit(archive, audited, progress=False)
    output = tmp_path / "bids"
    manifest = import_bidding(audited, output, max_games=100, progress=False)
    assert manifest["provenance"]["temporal_boundaries"] == [
        "2020-02-18T00:00:00",
        "2020-02-24T00:00:00",
    ]
    metadata = [
        json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()
    ]
    assert len(metadata) == sum(manifest["rows"].values())
    dataset = BiddingCorpus(output)
    partitions = {}
    for split in ("train", "val", "calibration", "test"):
        games = [g for g in metadata if g["split"] == split]
        states = np.concatenate([x for x, y in dataset.batches(split, 7)])
        labels = np.concatenate([y for x, y in dataset.batches(split, 7)])
        for g, x, y in zip(games, states, labels):
            actual = parse_record(lines[g["line"]])
            np.testing.assert_array_equal(
                x, bidding_state(actual.original[actual.soloist], actual.soloist, 2)
            )
            assert y == float(actual.won)
            assert not actual.contract.hand and actual.contract.kind != "null"
            assert g["group"] not in partitions or partitions[g["group"]] == split
            partitions[g["group"]] = split


def test_overbidding_losses_are_excluded_from_hand_labels(tmp_path):
    lines = []
    for i, line in enumerate(FIXTURE.read_text().splitlines(keepends=True)):
        date = (datetime(2020, 1, 1) + timedelta(days=i)).strftime(
            "%Y-%m-%d/%H:%M:%S/UTC"
        )
        line = re.sub(r"ID\[[^]]*\]", f"ID[{i}]", line)
        line = re.sub(r"SE\[[^]]*\]", f"SE[session-{i}]", line)
        line = re.sub(r"DT\[[^]]*\]", f"DT[{date}]", line)
        lines.append(line)
    original = parse_record(lines[0])
    assert original.score == -72 and original.bid == 18
    overbid = (
        lines[0].replace("ID[0]", "ID[overbid]").replace("SE[session-0]", "SE[overbid]")
    )
    overbid = overbid.replace("0 18 0 s", "0 40 0 s").replace("v:-72", "v:-96")
    assert parse_record(overbid).bid == 40
    archive = tmp_path / "input.sgf"
    archive.write_text("".join([*lines, overbid]))
    audit(archive, tmp_path / "audit", progress=False)
    manifest = import_bidding(
        tmp_path / "audit", tmp_path / "bidding", max_games=100, progress=False
    )
    assert manifest["provenance"]["skipped"]["overbid"] >= 1
    assert "iss:overbid" not in {
        json.loads(line)["id"]
        for line in (tmp_path / "bidding/games.jsonl").read_text().splitlines()
    }
