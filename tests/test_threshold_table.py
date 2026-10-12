import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
from scripts import sweep_bidding_table as sweep


def test_regret_means_unilateral_improvement_not_equal_scores():
    values = (0.0, 1.0)
    payoffs = np.empty((2, 2, 2, 3))
    for indices in np.ndindex((2, 2, 2)):
        payoffs[indices] = [indices[0], 2 * indices[1], 3 * indices[2]]
    # Equal payoffs at zero are not an equilibrium; unequal payoffs at one are.
    assert sweep.regrets(payoffs)[0, 0, 0].tolist() == [1, 2, 3]
    selected = sweep.choose_profile(payoffs, values)
    assert selected["thresholds"] == [1, 1, 1]
    assert selected["regrets"] == [0, 0, 0]
    assert len(selected["empirical_grid_equilibria"]) == 1


def test_response_profiles_hold_both_opponents_fixed_and_preserve_ties():
    current = (0.25, 0.5, 0.75)
    profiles = sweep.response_profiles(current)
    assert len(profiles) == 61
    assert all(sum(a != b for a, b in zip(p, current)) <= 1 for p in profiles)
    lookup = {p: {"seats": [{"wins": 12}] * 3} for p in profiles}
    answers = sweep.best_responses(current, lookup, "wins")
    assert [a["best_threshold"] for a in answers] == list(current)
    assert [a["gain"] for a in answers] == [0, 0, 0]


def test_threshold_grid():
    assert sweep.grid_values(0.25) == (0, 0.25, 0.5, 0.75, 1)
    assert len(sweep.grid_values(0.05)) == 21
    for value in (0, -1, 0.3, 0.01):
        with pytest.raises(ValueError):
            sweep.grid_values(value)


def test_joint_bidders_and_common_deals(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sweep, "PREDICTOR", SimpleNamespace(probability=lambda *args: 0.5)
    )
    baseline = sweep.run_case(None, 3, 42, tmp_path)
    single = sweep.run_case((0, 1, 1), 3, 42, tmp_path)
    passed = sweep.run_case((1, 1, 1), 3, 42, tmp_path)
    assert baseline["bidders"] == ["conservative"] * 3
    assert single["seats"][0]["solo_rounds"] == 3
    assert all(s["passed"] == 3 for s in passed["seats"])
    records = [
        json.loads((tmp_path / (sweep.case_name(t) + ".json")).read_text())
        for t in (None, (0, 1, 1), (1, 1, 1))
    ]
    assert [r["seed"] for r in records[0]["rounds"]] == [
        r["seed"] for r in records[1]["rounds"]
    ]
    for record in records:
        assert all(s["wins"] + s["losses"] + s["passed"] == 3 for s in record["seats"])


def test_decimal_slice_names_do_not_overwrite_each_other(tmp_path):
    pytest.importorskip("matplotlib")
    values = (0, 1)
    rows = [
        {"thresholds": (a, b, t), "seats": [{"game_points": 10 * (a - b)}] * 3}
        for a in values
        for b in values
        for t in (0.25, 0.75)
    ]
    for third in (0.25, 0.75):
        sweep.plot_surfaces(
            rows, values, values, third, tmp_path / f"surface-seat2-{third:.2f}", "Test"
        )
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "surface-seat2-0.25.png",
        "surface-seat2-0.75.png",
    }
    assert len(list(tmp_path.glob("*.pdf"))) == 2
