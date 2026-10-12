import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from scripts import sweep_bidding_threshold as sweep
from skat.agents.random import RandomAgent
from skat.config import MCTSConfig
from skat.game import SeatScore


def test_points_include_passes_without_crediting_defender_scores():
    score = SeatScore(
        solo_rounds=2, solo_wins=1, defence_rounds=1, defence_wins=1, game_points=-18
    )
    result = sweep.metrics(score, 5)
    assert result["points_per_round"] == -3.6
    assert result["win_rate"] == 2 / 3
    assert result["wins"] == 2 and result["losses"] == 1
    assert result["wins"] + result["losses"] + result["passed"] == 5
    assert result["game_points"] == -18
    assert result["solo_fraction"] == 2 / 5
    assert result["passed"] == 2
    run = {
        "rounds": [
            {"seed": 1, "result": {"soloist": 1, "score": 18}},
            {"seed": 2, "result": {"soloist": 1, "score": -36}},
            {"seed": 3, "result": {"soloist": 2, "score": -48}},
            {"seed": 4, "result": {"soloist": None, "score": 0}},
        ]
    }
    assert [r["official_points"] for r in sweep.point_outcomes(run, 1)] == [
        18,
        -36,
        0,
        0,
    ]


def test_sweep_changes_only_requested_bidder_and_recreates_agents(
    tmp_path, monkeypatch
):
    class Bidder:
        def __init__(self, path, *, device, threshold):
            self.threshold = threshold

        def probability(self, *args):
            return 0.5

    calls = []

    def make_agent(name, **kwargs):
        calls.append((name, kwargs))
        return RandomAgent(seed=kwargs["seed"]).configure_bidding(
            kwargs["bidding"], kwargs["bidder"]
        )

    monkeypatch.setattr(sweep, "NeuralBidder", Bidder)
    monkeypatch.setattr(sweep, "get_player", make_agent)
    args = SimpleNamespace(
        agents=["random", "policy", "mcts"],
        policy="shared.pt",
        policy0=None,
        policy1="seat1.pt",
        policy2=None,
        seed=42,
        rounds=2,
        bidding_seat=1,
        bidding_model="bid.pt",
        device="cpu",
        out=tmp_path,
    )
    baseline = sweep.play(args, MCTSConfig(4, 1), None, None)
    low = sweep.play(args, MCTSConfig(4, 1), None, 0.0)
    high = sweep.play(args, MCTSConfig(4, 1), None, 1.0)
    assert baseline["bidders"] == ["conservative"] * 3
    assert low["bidders"] == ["conservative", "neural (0)", "conservative"]
    assert high["bidders"] == ["conservative", "neural (1)", "conservative"]
    assert high["seats"][1]["solo_rounds"] == 0
    assert (
        [r["seed"] for r in baseline["rounds"]]
        == [r["seed"] for r in low["rounds"]]
        == [r["seed"] for r in high["rounds"]]
    )
    for index in range(0, len(calls), 3):
        assert [name for name, _ in calls[index : index + 3]] == args.agents
        assert [kwargs["seed"] for _, kwargs in calls[index : index + 3]] == [
            42,
            43,
            44,
        ]
        assert calls[index + 1][1]["policy"] == "seat1.pt"
        assert calls[index][1]["bidder"] is calls[index + 2][1]["bidder"] is None
    assert low["status"] == high["status"] == "complete"


def test_threshold_grid_includes_both_endpoints_without_accumulation():
    assert len(sweep.THRESHOLDS) == 21
    assert sweep.THRESHOLDS[0] == 0.0 and sweep.THRESHOLDS[-1] == 1.0
    assert sweep.THRESHOLDS[11] == 0.55


def test_missing_policy_fails_before_creating_results(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(SystemExit, match="2"):
        sweep.main(
            [
                "--agents",
                "random",
                "policy",
                "random",
                "--bidding-model",
                "missing.pt",
                "--out",
                str(output),
            ]
        )
    assert not output.exists()


def test_interruption_preserves_completed_rounds(tmp_path, monkeypatch):
    from skat import game

    original_round = game.Round
    calls = 0

    def fail_second_round(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted game")
        return original_round(**kwargs)

    monkeypatch.setattr(game, "Round", fail_second_round)
    args = SimpleNamespace(
        agents=["random"] * 3,
        policy=None,
        policy0=None,
        policy1=None,
        policy2=None,
        seed=42,
        rounds=3,
        bidding_seat=1,
        bidding_model=None,
        device="cpu",
        out=tmp_path,
    )
    with pytest.raises(RuntimeError, match="interrupted game"):
        sweep.play(args, MCTSConfig(4, 1), None, None)
    saved = json.loads((tmp_path / "baseline.json").read_text())
    assert saved["status"] == "interrupted"
    assert saved["completed_rounds"] == len(saved["rounds"]) == 1
    assert all(s["played"] + s["passed"] == 1 for s in saved["seats"])
