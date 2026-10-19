"""Short integration tests: real engine rounds, no training or strength runs."""

import threading
import time

import pytest

pytest.importorskip("flask")

from skat.rules import GameDeclaration  # noqa: E402
from skat.web import create_app  # noqa: E402
from skat.web.session import GameSession, SessionRegistry  # noqa: E402


@pytest.fixture
def table():
    registry = SessionRegistry()
    app = create_app(registry=registry)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, registry
    registry.close()


def config(kinds=("human", "human", "human"), **kwargs):
    return {"seats": [{"kind": kind} for kind in kinds], "speedMs": 0, **kwargs}


def create(table, value=None):
    client, registry = table
    response = client.post("/api/sessions", json=config() if value is None else value)
    assert response.status_code == 201, response.json
    key = response.json["id"]
    return key, registry.get(key)


def wait(session, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    state = session.get_snapshot()
    while not predicate(state):
        assert state["status"] != "error", state["error"]
        remaining = deadline - time.monotonic()
        assert remaining > 0, state
        state = session.get_snapshot(after=state["revision"], timeout=remaining)
    return state


def action(table, key, pending, value):
    return table[0].post(
        f"/api/sessions/{key}/actions",
        json={"decisionId": pending["id"], "value": value},
    )


def play_humans(
    table, key, session, *, pickup=True, declaration=None, play_action=None
):
    seen = set()
    deadline = time.monotonic() + 5
    while True:
        state = wait(
            session,
            lambda s: (
                s["status"] in {"finished", "passed"}
                or (s["pending"] and s["pending"]["id"] not in seen)
            ),
            timeout=max(0.01, deadline - time.monotonic()),
        )
        if state["status"] in {"finished", "passed"}:
            return state
        pending = state["pending"]
        seen.add(pending["id"])
        kind, options = pending["kind"], pending["options"]
        if kind == "bid":
            value = 18 if pending["seat"] == 0 and 18 in options else 0
        elif kind == "pickup":
            value = pickup
        elif kind == "press":
            value = options[:2]
        elif kind == "declare":
            value = declaration or next(
                d
                for d in options
                if d["kind"] == "grand"
                and not d["ouvert"]
                and not d["schneider_announced"]
            )
        elif kind == "play":
            value = (
                play_action
                if play_action in options
                else next(o for o in options if type(o) is int)
            )
            if value == play_action:
                play_action = None
        else:
            value = True
        response = action(table, key, pending, value)
        assert response.status_code == 200, response.json


def test_full_human_round_and_next_round_are_scored_once(table):
    key, session = create(table)
    state = play_humans(table, key, session)
    assert state["status"] == "finished"
    assert len(state["history"]) == 10
    assert sum(state["result"]["points"]) == 120
    assert sum(player["points"] for player in state["players"]) == 120
    assert state["scores"][0] == state["result"]["score"]
    assert len(state["records"]) == 1
    request = {"command": "next", "roundIndex": 0}
    assert (
        table[0].post(f"/api/sessions/{key}/control", json=request).status_code == 200
    )
    assert (
        table[0].post(f"/api/sessions/{key}/control", json=request).status_code == 409
    )
    state = wait(session, lambda s: s["roundIndex"] == 1 and s["pending"] is not None)
    assert state["dealer"] == 1
    assert len(state["records"]) == 1
    assert state["scores"][0] == state["records"][0]["result"]["score"]


def test_all_pass_and_long_poll_wakes_for_control(table):
    key, session = create(table)
    seen = set()
    for _ in range(3):
        state = wait(session, lambda s: s["pending"] and s["pending"]["id"] not in seen)
        seen.add(state["pending"]["id"])
        assert action(table, key, state["pending"], 0).status_code == 200
    state = wait(session, lambda s: s["status"] == "passed")
    assert state["scores"] == [0, 0, 0]
    assert len(state["records"]) == 1
    received = []
    waiter = threading.Thread(
        target=lambda: received.append(
            session.get_snapshot(after=state["revision"], timeout=1)
        )
    )
    waiter.start()
    session.control("reveal", True)
    waiter.join(2)
    assert received[0]["revision"] > state["revision"]


def test_invalid_and_duplicate_decisions_leave_round_unchanged(table):
    key, session = create(table)
    state = wait(session, lambda s: s["pending"] is not None)
    prompt = state["pending"]
    hand = tuple(session.round.player[prompt["seat"]].hand)
    for value in (True, "bad", [], {"bad": True}):
        assert action(table, key, prompt, value).status_code == 400
    assert tuple(session.round.player[prompt["seat"]].hand) == hand
    assert action(table, key, prompt, 0).status_code == 200
    assert action(table, key, prompt, 0).status_code == 409
    assert table[0].delete(f"/api/sessions/{key}").status_code == 204
    session.thread.join(2)
    assert not session.thread.is_alive()


def test_press_requires_two_distinct_owned_cards(table):
    key, session = create(table)
    seen = set()
    while True:
        state = wait(session, lambda s: s["pending"] and s["pending"]["id"] not in seen)
        pending = state["pending"]
        seen.add(pending["id"])
        if pending["kind"] == "press":
            break
        value = (
            True
            if pending["kind"] == "pickup"
            else 18
            if pending["seat"] == 0 and 18 in pending["options"]
            else 0
        )
        assert action(table, key, pending, value).status_code == 200
    assert len(session.round.player[0].hand) == 12
    owned = pending["options"]
    absent = next(i for i in range(32) if i not in owned)
    for value in (
        [owned[0]],
        [owned[0], owned[0]],
        [owned[0], absent],
        [True, owned[0]],
    ):
        assert action(table, key, pending, value).status_code == 400
        assert len(session.round.player[0].hand) == 12
    assert action(table, key, pending, owned[:2]).status_code == 200
    state = wait(session, lambda s: s["pending"] and s["pending"]["kind"] == "declare")
    assert len(session.round.player[0].hand) == 10
    assert state["lastAction"]["value"] is None  # no discard leak through action log
    invalid = next(
        d.copy()
        for d in state["pending"]["options"]
        if d["kind"] == "suit" and d["suit"] == 1
    )
    invalid["suit"] = True
    assert action(table, key, state["pending"], invalid).status_code == 400
    assert len(session.round.player[0].hand) == 10


@pytest.mark.parametrize(
    "declaration",
    [
        GameDeclaration("null"),
        GameDeclaration("null", hand=True, ouvert=True),
        GameDeclaration("suit", 2, hand=True, schneider_announced=True),
        GameDeclaration("grand", hand=True, ouvert=True),
    ],
)
def test_current_declarations_use_official_engine_results(table, declaration):
    from dataclasses import asdict

    key, session = create(table)
    state = play_humans(
        table,
        key,
        session,
        pickup=not declaration.hand,
        declaration=asdict(declaration),
    )
    assert state["result"]["declaration"] == asdict(declaration)
    assert state["scores"][0] == session.round.result.score


@pytest.mark.parametrize("special", ["concede", "claim0", "open"])
def test_early_actions_and_defender_confirmations(table, special):
    key, session = create(table)
    state = play_humans(table, key, session, play_action=special)
    assert state["result"]["reason"] in {"conceded", "claimed", "defender-open"}
    assert len(state["records"]) == 1


def test_observer_pause_step_reveal_and_hidden_information(table):
    key, session = create(table, config(("random",) * 3, paused=True))
    state = wait(session, lambda s: s["active"] is not None)
    assert all(not p["hand"] and p["handCount"] == 10 for p in state["players"])
    assert state["skat"] == []
    session.control("reveal", True)
    revealed = session.get_snapshot()
    assert all(len(p["hand"]) == 10 for p in revealed["players"])
    assert len(revealed["skat"]) == 2
    assert all(agent.delegate.state.skat == () for agent in session.agents)
    session.control("reveal", False)
    assert all(not p["hand"] for p in session.get_snapshot()["players"])
    session.control("step")
    stepped = wait(session, lambda s: s["lastAction"] is not None)
    assert stepped["paused"]
    session.control("resume")
    completed = wait(session, lambda s: s["status"] in {"finished", "passed"})
    assert completed["result"] == completed["records"][0]["result"]


def test_tiny_mcts_round_and_pool_cleanup(table):
    key, session = create(
        table,
        config(
            ("mcts", "random", "random"), simulations=2, determinizations=2, workers=2
        ),
    )
    state = wait(session, lambda s: s["status"] in {"finished", "passed"})
    assert state["error"] is None
    table[0].delete(f"/api/sessions/{key}")
    session.thread.join(3)
    assert not session.thread.is_alive()
    assert session.pool.closed


def test_validation_json_static_files_and_origin(table):
    client, _ = table
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/config").json["agents"] == [
        "human",
        "random",
        "mcts",
        "policy",
    ]
    for value in (
        None,
        [],
        {},
        config(workers=0),
        config(seed=True),
        config(device="bad"),
    ):
        assert client.post("/api/sessions", json=value).status_code in {400, 415}
    assert (
        client.post(
            "/api/sessions", json=config(), headers={"Origin": "https://other.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/sessions", json=config(), headers={"Host": "other.example"}
        ).status_code
        == 403
    )
    assert client.get("/api/sessions/missing").status_code == 404
    assert (
        client.post(
            "/api/sessions", json=config(("policy", "random", "random"))
        ).status_code
        == 400
    )


def test_bad_checkpoint_surfaces_loading_error(table):
    pytest.importorskip("torch")
    value = config(("policy", "random", "random"))
    value["seats"][0]["policy"] = "/nonexistent/policy.pt"
    key, session = create(table, value)
    state = wait(session, lambda s: s["status"] == "error")
    assert "FileNotFoundError" in state["error"]
    session.thread.join(3)
    assert session.pool.closed


def test_seeded_rounds_are_reproducible(table):
    _, first = create(table, config(("random",) * 3))
    _, second = create(table, config(("random",) * 3))
    a = wait(first, lambda s: s["status"] in {"finished", "passed"})
    b = wait(second, lambda s: s["status"] in {"finished", "passed"})
    assert a["history"] == b["history"]
    assert a["result"] == b["result"]


def test_session_config_is_detached_from_caller():
    value = config()
    session = GameSession(value)
    value["seats"][0]["kind"] = "random"
    assert session.config["seats"][0]["kind"] == "human"


def test_malformed_control_stays_a_client_error(table):
    key, session = create(table)
    for command in (None, [], {}):
        response = table[0].post(
            f"/api/sessions/{key}/control", json={"command": command}
        )
        assert response.status_code == 400
    assert session.get_snapshot()["status"] != "error"


def test_tiny_policy_checkpoint_plays_without_training(table, tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.distill.checkpoint import save_policy
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    model = CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    path = tmp_path / "policy.pt"
    save_policy(path, model)
    value = config(("policy", "random", "random"))
    value["seats"][0]["policy"] = str(path)
    _, session = create(table, value)
    state = wait(session, lambda s: s["status"] in {"finished", "passed"})
    assert state["error"] is None
    assert state["result"] == state["records"][0]["result"]
