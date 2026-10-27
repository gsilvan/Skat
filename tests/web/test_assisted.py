"""Manual execution uses real agents and the existing validated browser actions."""

import threading

import pytest

pytest.importorskip("flask")

from skat.web.agents import shown  # noqa: E402
from skat.web.session import GameSession, validate_config  # noqa: E402

from .test_web import action, config, create, play_humans, wait  # noqa: E402
from .test_web import table as table  # noqa: E402


def assisted_config(kinds=("random",) * 3, **kwargs):
    value = config(kinds, **kwargs)
    for seat in value["seats"]:
        seat["execution"] = "manual"
    return value


def accept_suggestions(table, key, session, choose=None):
    seen = set()
    for _ in range(120):
        state = wait(
            session,
            lambda s: (
                s["status"] in {"finished", "passed"}
                or (s["pending"] and s["pending"]["id"] not in seen)
            ),
        )
        if state["status"] in {"finished", "passed"}:
            return state
        pending = state["pending"]
        seen.add(pending["id"])
        value = pending["suggestion"] if choose is None else choose(pending)
        assert action(table, key, pending, value).status_code == 200
    pytest.fail("Round did not finish")


def test_execution_defaults_and_validation():
    value = validate_config(config(("human", "random", "mcts")))
    assert [seat["execution"] for seat in value["seats"]] == ["manual", "auto", "auto"]
    for kind, execution in [("human", "auto"), ("random", True), ("random", "bad")]:
        value = config((kind, "random", "random"))
        value["seats"][0]["execution"] = execution
        with pytest.raises(ValueError, match="Ausführung"):
            validate_config(value)
    value = assisted_config(("policy", "random", "random"))
    with pytest.raises(ValueError, match="policy"):
        validate_config(value)
    value = assisted_config()
    value["seats"][0]["bidding"] = "neural"
    with pytest.raises(ValueError, match="biddingModel"):
        validate_config(value)


def test_accepting_all_suggestions_preserves_automatic_round_and_next(table):
    _, automatic = create(table, config(("random",) * 3))
    expected = wait(automatic, lambda s: s["status"] in {"finished", "passed"})
    key, session = create(table, assisted_config())
    state = accept_suggestions(table, key, session)
    assert state["history"] == expected["history"]
    assert state["result"] == expected["result"]
    assert state["comparisons"] and all(row["matches"] for row in state["comparisons"])
    assert {row["kind"] for row in state["comparisons"]} == {
        "bid",
        "pickup",
        "press",
        "declare",
        "play",
    }
    assert len({row["decisionId"] for row in state["comparisons"]}) == len(
        state["comparisons"]
    )
    for stats in state["comparisonStats"]:
        rows = [row for row in state["comparisons"] if row["seat"] == stats["seat"]]
        assert stats["matches"] == stats["total"] == len(rows)
        assert sum(counter["total"] for counter in stats["byKind"].values()) == len(
            rows
        )
    previous = state["comparisons"]
    automatic.control("next", round_index=0)
    expected = wait(
        automatic,
        lambda s: s["roundIndex"] == 1 and s["status"] in {"finished", "passed"},
    )
    session.control("next", round_index=0)
    state = wait(session, lambda s: s["roundIndex"] == 1 and s["pending"] is not None)
    assert state["comparisons"] == previous
    completed = accept_suggestions(table, key, session)
    assert completed["history"] == expected["history"]
    assert completed["result"] == expected["result"]
    assert completed["comparisons"][: len(previous)] == previous
    assert any(row["roundIndex"] == 1 for row in completed["comparisons"])


def test_overrides_use_human_actions_and_compare_discard_pairs_as_sets(table):
    key, session = create(table, assisted_config())
    reversed_pair = None

    def choose(pending):
        nonlocal reversed_pair
        kind, options = pending["kind"], pending["options"]
        if kind == "bid":
            return 18 if pending["seat"] == 0 and 18 in options else 0
        if kind == "pickup":
            return True
        if kind == "press":
            reversed_pair = pending["suggestion"][::-1]
            return reversed_pair
        if kind == "declare":
            return next(d for d in options if d["kind"] == "grand" and not d["ouvert"])
        if kind == "play":
            return next(
                (v for v in options if type(v) is int and v != pending["suggestion"]),
                pending["suggestion"],
            )
        return pending["suggestion"]

    state = accept_suggestions(table, key, session, choose)
    assert state["soloist"] == 0 and len(state["history"]) == 10
    press = next(row for row in state["comparisons"] if row["kind"] == "press")
    assert press["actual"] == reversed_pair and press["matches"]
    assert {c.np_index for c in session.round.skat} == set(reversed_pair)
    plays = [row for row in state["comparisons"] if row["kind"] == "play"]
    assert len(plays) == 30 and any(not row["matches"] for row in plays)
    actual_cards = [
        card["card"]["id"] for trick in state["history"] for card in trick["cards"]
    ]
    assert actual_cards == [row["actual"] for row in plays]


def test_pending_is_stable_visible_and_validated_even_while_paused(table):
    value = assisted_config(("random", "human", "random"), paused=True, dealer=1)
    value["seats"][2]["execution"] = "auto"
    key, session = create(table, value)
    state = wait(session, lambda s: s["pending"] is not None)
    pending = state["pending"]
    assert pending["seat"] == 0 and "suggestion" in pending
    assert [len(player["hand"]) for player in state["players"]] == [10, 10, 0]
    assert state["skat"] == []
    assert session.agents[0].delegate.state.skat == ()
    assert not any(session.agents[0].delegate.state.exposed_hands)
    original_hand = tuple(session.round.player[0].hand)
    rng_state = session.agents[0].delegate.rng.getstate()
    for _ in range(2):
        response = table[0].get(f"/api/sessions/{key}")
        assert response.json["pending"] == pending
    session.control("reveal", True)
    assert all(len(p["hand"]) == 10 for p in session.get_snapshot()["players"])
    assert session.agents[0].delegate.state.skat == ()
    session.control("reveal", False)
    assert session.agents[0].delegate.rng.getstate() == rng_state
    assert action(table, key, pending, True).status_code == 400
    assert session.get_snapshot()["comparisons"] == []
    assert tuple(session.round.player[0].hand) == original_hand
    assert action(table, key, pending, pending["suggestion"]).status_code == 200
    assert action(table, key, pending, pending["suggestion"]).status_code == 409
    state = wait(session, lambda s: len(s["comparisons"]) == 1)
    assert state["comparisons"][0]["decisionId"] == pending["id"]
    assert state["comparisonStats"][0]["total"] == 1
    assert action(table, key, pending, 0).status_code == 409


@pytest.mark.parametrize("special", ["concede", "claim0", "open"])
def test_special_actions_and_confirmation_overrides(table, special):
    key, session = create(table, assisted_config())
    state = play_humans(table, key, session, play_action=special)
    assert state["result"]["reason"] in {"conceded", "claimed", "defender-open"}
    assert any(
        row["actual"] == special and not row["matches"] for row in state["comparisons"]
    )
    for row in state["comparisons"]:
        if row["kind"].startswith("confirm_"):
            assert row["suggestion"] is False
            assert row["actual"] is True and not row["matches"]
    if special == "claim0":
        assert any(row["kind"] == "confirm_claim" for row in state["comparisons"])


def test_hand_override_skips_press_and_suggests_hand_declaration(table):
    key, session = create(table, assisted_config())
    state = play_humans(table, key, session, pickup=False)
    assert state["declaration"]["hand"]
    assert not any(row["kind"] == "press" for row in state["comparisons"])
    pickup = next(row for row in state["comparisons"] if row["kind"] == "pickup")
    assert pickup["suggestion"] is True and pickup["actual"] is False
    declaration = next(row for row in state["comparisons"] if row["kind"] == "declare")
    assert declaration["suggestion"]["hand"]


@pytest.mark.parametrize("neural", [False, True])
def test_changed_discards_update_next_suggestion_and_visible_skat(
    table, monkeypatch, neural
):
    import numpy as np

    from skat.agents.random import RandomAgent

    class KeptCardScorer:
        def __init__(self):
            self.batches = []

        def probabilities(self, features):
            values = np.stack(features)
            self.batches.append(values.copy())
            target = np.flatnonzero(values[0, :32] + values[0, 32:64])[0]
            return values[:, 103] * values[:, target] + values[:, 104] * (
                1 - values[:, target]
            )

    scorer = KeptCardScorer()
    if neural:
        pytest.importorskip("torch")
        make_agents = GameSession.make_agents

        def configured(session):
            agents = make_agents(session)
            agents[0].delegate.configure_declaration("neural", scorer)
            return agents

        monkeypatch.setattr(GameSession, "make_agents", configured)
    # Both opponents pass; the human can override their own suggested pass.
    monkeypatch.setattr(RandomAgent, "bid", lambda *args: 0)
    value = config(("random",) * 3)
    value["seats"][0]["execution"] = "manual"
    key, session = create(table, value)
    actual, original_skat, initial_kind = None, None, None

    def choose(pending):
        nonlocal actual, original_skat, initial_kind
        if pending["kind"] == "bid":
            return 18
        if pending["kind"] == "press":
            state = session.get_snapshot()
            original_skat = [card["id"] for card in state["skat"]]
            assert len(original_skat) == 2 and len(state["players"][0]["hand"]) == 12
            assert not state["players"][1]["hand"] and not state["players"][2]["hand"]
            initial_kind = session.agents[
                0
            ].delegate._declaration_choice.game.declaration.kind
            owned = sorted(pending["options"])
            if neural and initial_kind == "null":
                actual = owned[1:3]
            else:
                actual = owned[:2]
            if set(actual) == set(pending["suggestion"]):
                actual = [actual[0], owned[-1]]
            assert set(actual) != set(pending["suggestion"])
            return actual
        if pending["kind"] == "declare":
            state = session.get_snapshot()
            assert {card["id"] for card in state["skat"]} == set(actual)
            assert len(state["players"][0]["hand"]) == 10
            assert not set(actual) & {
                card["id"] for card in state["players"][0]["hand"]
            }
            if neural:
                fixed = scorer.batches[-1]
                assert len(scorer.batches) == 2 and len(fixed) <= 6
                assert set(np.flatnonzero(fixed[0, 32:64])) == set(actual)
                assert set(np.flatnonzero(fixed[0, 64:96])) == set(original_skat)
                assert pending["suggestion"]["kind"] != initial_kind
        return pending["suggestion"]

    state = accept_suggestions(table, key, session, choose)
    pressed = next(row for row in state["comparisons"] if row["kind"] == "press")
    assert not pressed["matches"]
    assert len(state["comparisonStats"]) == 1


def test_manual_mcts_uses_pool_and_finishes_without_recomputation(table):
    key, session = create(
        table,
        assisted_config(
            ("mcts", "random", "random"), simulations=2, determinizations=2, workers=2
        ),
    )
    state = accept_suggestions(table, key, session)
    assert len(state["history"]) == 10
    assert session.agents[0].delegate.search_pool is session.pool
    table[0].delete(f"/api/sessions/{key}")
    session.thread.join(3)
    assert not session.thread.is_alive() and session.pool.closed


def test_manual_policy_with_tiny_checkpoint(table, tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.distill.checkpoint import save_policy
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    path = tmp_path / "policy.pt"
    save_policy(
        path, CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    )
    value = assisted_config(("policy", "random", "random"))
    value["seats"][0]["policy"] = str(path)
    key, session = create(table, value)
    state = accept_suggestions(table, key, session)
    assert any(
        row["seat"] == 0 and row["kind"] == "play" for row in state["comparisons"]
    )
    assert all(row["matches"] for row in state["comparisons"])


def test_stop_during_suggestion_closes_pool_and_does_not_publish_pending(
    table, monkeypatch
):
    from skat.agents.random import RandomAgent

    started, release = threading.Event(), threading.Event()
    original = RandomAgent.bid

    def blocked(self, current_bid, offer=None):
        started.set()
        assert release.wait(3)
        return original(self, current_bid, offer)

    monkeypatch.setattr(RandomAgent, "bid", blocked)
    key, session = create(table, assisted_config())
    try:
        assert started.wait(3)
        assert session.get_snapshot()["active"]["kind"] == "bid"
        table[0].delete(f"/api/sessions/{key}")
    finally:
        release.set()
        session.thread.join(3)
    assert not session.thread.is_alive() and session.pool.closed
    assert session.get_snapshot()["pending"] is None
    assert session.get_snapshot()["comparisons"] == []


def test_invalid_suggestion_surfaces_session_error(table, monkeypatch):
    from skat.agents.random import RandomAgent

    monkeypatch.setattr(RandomAgent, "bid", lambda *args: True)
    _, session = create(table, assisted_config())
    state = wait(session, lambda s: s["status"] == "error")
    assert "Agentenvorschlag" in state["error"]
    session.thread.join(3)
    assert session.pool.closed


def test_special_agent_suggestions_use_browser_action_format():
    from skat.rules import Claim, Concede, OpenHand

    assert shown(Concede()) == "concede"
    assert shown(OpenHand()) == "open"
    assert shown(Claim(2)) == "claim2"
