"""Streaming audit/index and deterministic, leakage-safe demonstration import."""

import bz2
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from iss import require_sgf_support
from iss.protocol import RejectedRecord, parse_record, replay
from skat.agents.distill.demonstrations import DemonstrationWriter

AUDIT_FORMAT = "iss-audit-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def records(path, *, progress=True):
    path = Path(path)
    with path.open("rb") as raw:
        stream = (
            bz2.open(raw, "rt", encoding="utf-8")
            if path.suffix == ".bz2"
            else open(raw.fileno(), "rt", encoding="utf-8", closefd=False)
        )
        with (
            stream,
            tqdm(
                total=path.stat().st_size,
                desc="ISS archive",
                unit="B",
                unit_scale=True,
                disable=not progress,
            ) as bar,
        ):

            def lines():
                for index, line in enumerate(stream):
                    bar.update(raw.tell() - bar.n)
                    if line.strip():
                        yield index, line
                bar.update(path.stat().st_size - bar.n)

            yield lines()


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-16384")  # 16 MiB regardless of archive size
    db.execute("PRAGMA temp_store=FILE")
    return db


class Groups:
    """Disk-backed union/find of session and deal identities, including dates."""

    def __init__(self, db, readonly=False):
        self.db = db
        self.readonly = readonly

    def root(self, name):
        row = self.db.execute(
            "SELECT parent FROM groups WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            if self.readonly:
                raise ValueError("missing audit group")
            self.db.execute(
                "INSERT INTO groups VALUES (?, ?, 0, NULL, NULL)", (name, name)
            )
            return name
        path = []
        while row[0] != name:
            path.append(name)
            name = row[0]
            row = self.db.execute(
                "SELECT parent FROM groups WHERE name=?", (name,)
            ).fetchone()
        if not self.readonly:
            self.db.executemany(
                "UPDATE groups SET parent=? WHERE name=?", ((name, p) for p in path)
            )
        return name

    def add(self, session, deal, date):
        a, b = self.root(session), self.root("deal:" + deal)
        if a != b:
            ra = self.db.execute(
                "SELECT rank, first_date, last_date FROM groups WHERE name=?", (a,)
            ).fetchone()
            rb = self.db.execute(
                "SELECT rank, first_date, last_date FROM groups WHERE name=?", (b,)
            ).fetchone()
            if ra[0] < rb[0]:
                a, b, ra, rb = b, a, rb, ra
            self.db.execute("UPDATE groups SET parent=? WHERE name=?", (a, b))
            dates = [d for d in (*ra[1:], *rb[1:]) if d]
            self.db.execute(
                "UPDATE groups SET rank=?, first_date=?, last_date=? WHERE name=?",
                (
                    ra[0] + int(ra[0] == rb[0]),
                    min(dates, default=date),
                    max(dates, default=date),
                    a,
                ),
            )
        self.db.execute(
            "UPDATE groups SET first_date=min(coalesce(first_date, ?), ?), "
            "last_date=max(coalesce(last_date, ?), ?) WHERE name=?",
            (date, date, date, date, a),
        )
        return a


def audit(archive, output, *, progress=True):
    require_sgf_support()
    archive, output = Path(archive).resolve(), Path(output)
    if not archive.is_file():
        raise ValueError("archive does not exist")
    output.mkdir(parents=True, exist_ok=False)
    digest = sha256(archive)
    counts, actors, contracts, years = Counter(), Counter(), Counter(), Counter()
    db = connect(output / "index.sqlite")
    try:
        db.executescript("""
            CREATE TABLE records (
                line INTEGER PRIMARY KEY, id TEXT UNIQUE, session TEXT, date TEXT,
                deal TEXT, group_id TEXT, metadata TEXT, raw_hash TEXT,
                reason TEXT, decisions INTEGER
            );
            CREATE TABLE groups (
                name TEXT PRIMARY KEY, parent TEXT, rank INTEGER,
                first_date TEXT, last_date TEXT
            );
        """)
        groups = Groups(db)
        with records(archive, progress=progress) as lines:
            for index, line in lines:
                counts["records"] += 1
                try:
                    game = parse_record(line)
                    decisions = replay(game, capture=False)
                    raw_hash = hashlib.sha256(line.strip().encode()).hexdigest()
                    prior = db.execute(
                        "SELECT line, raw_hash, reason FROM records WHERE id=?",
                        (game.identifier,),
                    ).fetchone()
                    if prior:
                        if prior[1] != raw_hash:
                            # Conflicting IDs invalidate the original as well.
                            db.execute(
                                "UPDATE records SET reason='conflicting_id' WHERE line=?",
                                (prior[0],),
                            )
                            raise RejectedRecord("conflicting_id")
                        raise RejectedRecord("duplicate_id")
                    group = groups.add(game.session, game.deal_hash, game.date)
                    db.execute(
                        "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                        (
                            index,
                            game.identifier,
                            game.session,
                            game.date,
                            game.deal_hash,
                            group,
                            json.dumps(game.metadata),
                            raw_hash,
                            decisions,
                        ),
                    )
                    counts["validated"] += 1
                    actors.update(game.players)
                    contracts[str(game.contract)] += 1
                    years[game.date[:4]] += 1
                except RejectedRecord as error:
                    counts[error.reason] += 1
                    db.execute(
                        "INSERT INTO records (line, reason) VALUES (?, ?)",
                        (index, error.reason),
                    )
                if counts["records"] % 1000 == 0:
                    db.commit()
        db.executescript(
            "CREATE INDEX record_dates ON records(date); CREATE INDEX record_groups ON records(group_id);"
        )
        db.commit()
        eligible = db.execute(
            "SELECT count(*) FROM records WHERE reason IS NULL"
        ).fetchone()[0]
        report = dict(
            format=AUDIT_FORMAT,
            archive=str(archive),
            archive_sha256=digest,
            counts=dict(counts),
            eligible_games=eligible,
            actors=dict(actors),
            contracts=dict(contracts),
            years=dict(years),
            complete=True,
            identity_policy="all actors unknown unless independently verified",
        )
        write_json(output / "audit.json", report)
        return report
    finally:
        db.close()


def select_games(
    audit_directory,
    output,
    *,
    max_games,
    seed,
    predicate="",
    validation_halves=False,
    archive_override=None,
):
    """Select immutable game groups once for card-play or bidding import.

    predicate is internal SQL, never supplied by CLI input. Time boundaries and
    group extents always use the full audit, regardless of the eligibility filter.
    """
    if type(max_games) is not int or max_games < 3:
        raise ValueError("max-games must be at least three")
    directory, output = Path(audit_directory), Path(output)
    report = json.loads((directory / "audit.json").read_text())
    if report.get("format") != AUDIT_FORMAT or not report.get("complete"):
        raise ValueError("incomplete or incompatible audit")
    archive = (
        Path(archive_override)
        if archive_override is not None
        else Path(report["archive"])
    )
    if sha256(archive) != report["archive_sha256"]:
        raise ValueError("archive changed since audit")
    # Keep only the selected index in the corpus; the full audit stays read-only.
    if output.exists():
        raise ValueError("use a new corpus directory")
    output.mkdir(parents=True)
    source = sqlite3.connect(
        f"file:{(directory / 'index.sqlite').resolve()}?mode=ro", uri=True
    )
    db = connect(output / "index.sqlite")
    try:
        total = source.execute(
            "SELECT count(*) FROM records WHERE reason IS NULL AND decisions>0"
        ).fetchone()[0]
        if total < 3:
            raise ValueError("at least three valid games with decisions required")

        # Time boundaries come from the FULL eligible archive, not the pilot.
        def boundary(fraction):
            return source.execute(
                "SELECT date FROM records WHERE reason IS NULL AND decisions>0 "
                "ORDER BY date, line LIMIT 1 OFFSET ?",
                (min(total - 1, int(total * fraction)),),
            ).fetchone()[0]

        first, second = boundary(0.8), boundary(0.9)
        source.create_function(
            "sample_key",
            1,
            lambda identifier: hashlib.sha256(
                f"{seed}:{identifier}".encode()
            ).hexdigest(),
        )
        db.execute(
            "CREATE TABLE selection (line INTEGER PRIMARY KEY, split TEXT, group_id TEXT)"
        )
        groups = Groups(source, readonly=True)
        selected = Counter()
        excluded = 0
        for line, group in source.execute(
            "SELECT line, group_id FROM records WHERE reason IS NULL AND decisions>0"
            + predicate
            + " ORDER BY sample_key(id)"
        ):
            root = groups.root(group)
            start, end = source.execute(
                "SELECT first_date, last_date FROM groups WHERE name=?", (root,)
            ).fetchone()
            split = (
                "train"
                if end < first
                else "val"
                if start >= first and end < second
                else "test"
                if start >= second
                else None
            )
            if validation_halves and split == "val":
                split = (
                    "val"
                    if int(
                        hashlib.sha256(
                            f"{seed}:calibration:{root}".encode()
                        ).hexdigest(),
                        16,
                    )
                    % 2
                    else "calibration"
                )
            if split is None:
                excluded += 1
                continue
            if sum(selected.values()) < max_games:
                db.execute(
                    "INSERT INTO selection VALUES (?, ?, ?)", (line, split, root)
                )
                selected[split] += 1
        db.commit()
        splits = (
            ("train", "val", "calibration", "test")
            if validation_halves
            else ("train", "val", "test")
        )
        if any(selected[split] == 0 for split in splits):
            raise ValueError(
                "temporal/session/deal split has an empty partition; increase max-games or inspect audit"
            )
        return archive, dict(
            source="ISS",
            archive_sha256=report["archive_sha256"],
            seed=seed,
            requested_games=max_games,
            selected_games=dict(selected),
            excluded_boundary_games=excluded,
            temporal_boundaries=[first, second],
            sampling="sha256(seed:id), whole games; no strength/outcome weighting",
        )
    finally:
        source.close()
        db.close()


def import_corpus(audit_directory, output, *, max_games=100000, seed=7, progress=True):
    require_sgf_support()
    archive, provenance = select_games(
        audit_directory, output, max_games=max_games, seed=seed
    )
    output = Path(output)
    writer = DemonstrationWriter(output)
    with connect(output / "index.sqlite") as db:
        targets = iter(db.execute("SELECT line, split FROM selection ORDER BY line"))
        target = next(targets, None)
        with records(archive, progress=progress) as lines:
            for index, line in lines:
                if target is None:
                    break
                if index != target[0]:
                    continue
                game = parse_record(line)
                writer.add_game(index, target[1], game.metadata, replay(game))
                target = next(targets, None)
        if target is not None:
            raise ValueError("archive ended before selected record")
    return writer.finish(provenance)
