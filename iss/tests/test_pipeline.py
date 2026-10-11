import bz2
import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pysgf")

from iss.pipeline import audit, import_corpus
from iss.protocol import RejectedRecord, ReplayRound, declaration, parse_record, replay
from skat.agents.distill.demonstrations import Demonstrations
from skat.models.observation import INPUT_SIZE

FIXTURE = Path(__file__).parent / "fixtures" / "card-games.sgf"


def fixture_lines():
    return FIXTURE.read_text().splitlines(keepends=True)


def dated_lines():
    lines = []
    for i, line in enumerate(fixture_lines()):
        line = re.sub(r"ID\[[^]]*\]", f"ID[{i}]", line)
        line = re.sub(r"SE\[[^]]*\]", f"SE[session-{i}]", line)
        line = re.sub(r"DT\[[^]]*\]", f"DT[2020-01-{i + 1:02d}/00:00:00/UTC]", line)
        lines.append(line)
    return lines


@pytest.mark.parametrize("line", fixture_lines())
def test_real_games_replay_and_produce_only_legal_nonforced_actions(line):
    game = parse_record(line)
    rows = replay(game)
    assert len(rows) == replay(game, capture=False)
    for state, mask, choice, seat in rows:
        assert state.shape == (INPUT_SIZE,)
        assert mask.sum() > 1 and mask[choice]
        assert state[choice] == 1  # still in hand before choosing
        assert 0 <= seat <= 2


@pytest.mark.parametrize(
    "token,flags",
    [
        ("GHS", (True, False, True, False)),
        ("GHZ", (True, False, True, True)),
        ("GO", (True, True, True, True)),
        ("NOH", (True, True, False, False)),
        ("NO", (False, True, False, False)),
    ],
)
def test_complete_declaration_flags(token, flags):
    d = declaration(token)
    assert (d.hand, d.ouvert, d.schneider_announced, d.schwarz_announced) == flags


@pytest.mark.parametrize("token", ["GS", "NZ", "NS", "GHBOGUS"])
def test_unsupported_flags_are_rejected(token):
    with pytest.raises(RejectedRecord):
        declaration(token)


def test_pipe_separator_and_sgf_escaping():
    line = fixture_lines()[0]
    game = parse_record(line)
    escaped = re.sub(r"P0\[[^]]*\]", lambda m: r"P0[a\]b]", line)
    altered = parse_record(escaped.replace(".", "|"))
    assert altered.players[0] == "a]b"
    assert altered.original == game.original
    assert replay(altered)


def test_bidding_actor_and_special_events_are_not_skipped():
    line = fixture_lines()[0]
    with pytest.raises(RejectedRecord, match="bidding_turn"):
        parse_record(line.replace("1 18 0 p", "2 18 0 p", 1))
    with pytest.raises(RejectedRecord, match="unsupported_event"):
        parse_record(line.replace("0 SQ", "0 SC", 1))


def test_replay_rejects_wrong_actor_illegal_card_and_wrong_result():
    game = parse_record(fixture_lines()[0])
    seat, chosen = game.actions[0]
    with pytest.raises(RejectedRecord, match="card_turn"):
        replay(replace(game, actions=(((seat + 1) % 3, chosen), *game.actions[1:])))
    with pytest.raises(RejectedRecord, match="illegal_card"):
        replay(
            replace(
                game, actions=((seat, game.hands[(seat + 1) % 3][0]), *game.actions[1:])
            )
        )
    with pytest.raises(RejectedRecord, match="result_or_scoring"):
        replay(replace(game, score=game.score + 1))
    with pytest.raises(RejectedRecord, match="incomplete_card_play"):
        replay(replace(game, actions=game.actions[:3]))


def make_round(game, hands=None):
    r = ReplayRound(
        dealer=2,
        start=False,
        skip_bidding=True,
        solo_player_id=game.soloist,
        declare_game=game.contract,
        hand_game=True,
        replay=True,
        initial_cards=[*(hands or game.hands), game.skat],
    )
    r.highest_bid = game.bid
    r.prepare()
    return r


def test_observer_does_not_see_hidden_hands_or_hand_skat():
    game = parse_record(fixture_lines()[0])  # clubs hand, no exposed cards
    hands = [list(h) for h in game.hands]
    hands[1][0], hands[2][0] = hands[2][0], hands[1][0]
    np.testing.assert_array_equal(
        make_round(game).get_state(0), make_round(game, hands).get_state(0)
    )
    assert not make_round(game).get_state(game.soloist)[-32:].any()


def test_pickup_skat_known_only_to_soloist_and_ouvert_is_public():
    games = [parse_record(line) for line in fixture_lines()]
    pickup = next(g for g in games if not g.contract.hand)
    r = make_round(pickup)
    assert r.get_state(pickup.soloist)[-32:].sum() == 2
    assert r.get_state((pickup.soloist + 1) % 3)[-32:].sum() == 0
    ouvert = next(g for g in games if g.contract.ouvert)
    assert make_round(ouvert).exposed == {ouvert.soloist}


def test_streaming_audit_import_dedup_and_temporal_grouping(tmp_path):
    lines = dated_lines()
    # One repeated deal across the time boundary must exclude both records.
    repeated = (
        lines[0]
        .replace("ID[0]", "ID[repeated]")
        .replace("SE[session-0]", "SE[repeated]")
    )
    repeated = re.sub(r"DT\[[^]]*\]", "DT[2020-02-01/00:00:00/UTC]", repeated)
    archive = tmp_path / "games.bz2"
    with bz2.open(archive, "wt") as f:
        f.write("".join([*lines, lines[1], repeated]))
    directory = tmp_path / "audit"
    report = audit(archive, directory, progress=False)
    assert report["counts"]["duplicate_id"] == 1
    corpus = tmp_path / "corpus"
    manifest = import_corpus(directory, corpus, max_games=100, progress=False)
    assert manifest["provenance"]["excluded_boundary_games"] >= 2
    games = [
        json.loads(line) for line in (corpus / "games.jsonl").read_text().splitlines()
    ]
    assert "iss:0" not in {g["id"] for g in games}
    assert "iss:repeated" not in {g["id"] for g in games}
    with sqlite3.connect(corpus / "index.sqlite") as db:
        assert not db.execute(
            "SELECT group_id FROM selection GROUP BY group_id HAVING count(DISTINCT split)>1"
        ).fetchall()
    dataset = Demonstrations(corpus)
    assert all(dataset.manifest["rows"][s] for s in ("train", "val", "test"))
    batches = list(dataset.batches("train", 7, seed=1))
    assert all(len(b["choices"]) <= 7 for b in batches)
    assert all(
        np.all(b["masks"][np.arange(len(b["choices"])), b["choices"]]) for b in batches
    )
    with pytest.raises(ValueError, match="new corpus"):
        import_corpus(directory, corpus, progress=False)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since audit"):
        import_corpus(directory, tmp_path / "changed", progress=False)


def test_conflicting_ids_reject_both_records(tmp_path):
    lines = dated_lines()
    duplicate = lines[0].replace("P0[jeff]", "P0[another]")
    # Guaranteed text difference without changing game legality.
    duplicate = duplicate.replace("CO[]", "CO[conflict]")
    archive = tmp_path / "games.sgf"
    archive.write_text("".join([*lines, duplicate]))
    report = audit(archive, tmp_path / "audit", progress=False)
    assert report["counts"]["conflicting_id"] == 1
    with sqlite3.connect(tmp_path / "audit" / "index.sqlite") as db:
        assert (
            db.execute("SELECT reason FROM records WHERE id='iss:0'").fetchone()[0]
            == "conflicting_id"
        )


def test_iss_train_checkpoint_and_live_play(tmp_path):
    pytest.importorskip("torch")
    from skat.agents.distill.checkpoint import load_policy
    from skat.agents.distill.train import train
    from skat.agents.policy import PolicyAgent
    from skat.agents.random import RandomAgent
    from skat.config import TrainConfig
    from skat.game import Tournament

    archive = tmp_path / "games.sgf"
    archive.write_text("".join(dated_lines()))
    audit(archive, tmp_path / "audit", progress=False)
    corpus = tmp_path / "corpus"
    import_corpus(tmp_path / "audit", corpus, max_games=100, progress=False)
    path = tmp_path / "policy.pt"
    report = train(
        corpus,
        path,
        config=TrainConfig(epochs=1, batch_size=128, d_model=16, nhead=2, num_layers=1),
    )
    assert report["target_source"] == "iss-action-v1" and report["test_rows"] > 0
    assert report["test"]["breakdown_test"]["defence"]["rows"] > 0
    _, metadata = load_policy(path)
    assert metadata["test"] == report["test"]
    tournament = Tournament(
        2, agents=[PolicyAgent(path), RandomAgent(1), RandomAgent(2)], seed=1
    )
    tournament.start()
    assert len(tournament.round_records) == 2
    shard = next(corpus.glob("train-*/choices.npy"))
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        Demonstrations(corpus)


@pytest.mark.parametrize("line", ["garbage", "(;GM[Skat]", "(;GM[Skat]) trailing"])
def test_malformed_sgf_has_a_rejection_reason(line):
    with pytest.raises(RejectedRecord):
        parse_record(line)


def test_deal_grouping_is_invariant_to_seat_rotation_and_card_order():
    game = parse_record(fixture_lines()[0])
    original = (
        *game.original[1:3],
        game.original[0],
        tuple(reversed(game.original[3])),
    )
    rotated = replace(game, original=tuple(tuple(reversed(h)) for h in original))
    assert rotated.deal_hash == game.deal_hash
