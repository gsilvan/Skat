from __future__ import annotations

import time

from skat.web import create_app


def wait_for(client, session_id, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get(f"/api/sessions/{session_id}").get_json()
        if predicate(state):
            return state
        time.sleep(0.01)
    raise AssertionError(f"session did not reach expected state: {state}")


def test_random_observer_round_completes_and_serializes_the_table():
    client = create_app().test_client()
    response = client.post(
        "/api/sessions",
        json={
            "seats": [{"kind": "random"}] * 3,
            "seed": 42,
            "speedMs": 0,
            "reveal": True,
        },
    )
    assert response.status_code == 201
    session_id = response.get_json()["id"]
    state = wait_for(
        client,
        session_id,
        lambda value: value["status"] in {"finished", "passed", "error"},
    )
    assert state["status"] == "finished"
    assert state["error"] is None
    assert len(state["history"]) == 10
    assert sum(player["handCount"] for player in state["players"]) == 0
    assert state["result"]["soloistWon"] is False


def test_human_prompt_rejects_illegal_actions():
    client = create_app().test_client()
    response = client.post(
        "/api/sessions",
        json={
            "seats": [
                {"kind": "human", "label": "You"},
                {"kind": "random"},
                {"kind": "random"},
            ],
            "seed": 7,
            "speedMs": 0,
        },
    )
    session_id = response.get_json()["id"]
    state = wait_for(client, session_id, lambda value: value["pending"] is not None)
    illegal = client.post(
        f"/api/sessions/{session_id}/actions", json={"value": "not-an-option"}
    )
    assert illegal.status_code == 400
    accepted = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"value": state["pending"]["options"][0]},
    )
    assert accepted.status_code == 200
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
