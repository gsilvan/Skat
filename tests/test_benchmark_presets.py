import copy
import hashlib

import pytest

from scripts.benchmark_presets import (
    paired_delta,
    play_deal,
    report,
    validate_outcomes,
    verify_models,
)
from skat.agents.random import RandomAgent


def configuration():
    return dict(
        seeds=[71],
        deals_per_seed=2,
        candidates=["best", "all-neural"],
        panels=["random", "best"],
    )


def factory():
    agents = {}

    def get(profile, seat):
        return agents.setdefault((profile, seat), RandomAgent())

    return get


def test_checkpoint_changes_invalidate_a_benchmark(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"frozen weights")
    config = dict(
        models={"policy": str(checkpoint)},
        checkpoint_sha256={
            "policy": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        },
    )
    verify_models(config)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoints changed"):
        verify_models(config)


def without_timing(rows):
    return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]


def test_each_deal_is_shared_by_all_profiles_panels_and_seats():
    config = configuration()
    first = play_deal(config, 71, 0, factory())
    assert len(first) == 12
    assert len({r["deal_code"] for r in first}) == 1
    assert {r["seat"] for r in first} == {0, 1, 2}
    for row in first:
        assert row["lineup"][row["seat"]] == row["agent"]
        assert all(
            profile == row["panel"]
            for s, profile in enumerate(row["lineup"])
            if s != row["seat"]
        )
        assert row["official_points"] == row["seat_points"][row["seat"]]
    second = play_deal(config, 71, 1, factory())
    assert first[0]["deal_code"] != second[0]["deal_code"]
    validate_outcomes(config, first + second)
    with pytest.raises(ValueError, match="missing, duplicate"):
        validate_outcomes(config, first + second[:-1])
    with pytest.raises(ValueError, match="missing, duplicate"):
        validate_outcomes(config, first + second + first[:1])


def test_results_do_not_depend_on_worker_chunk_or_profile_order():
    config = configuration()
    get = factory()
    play_deal(config, 71, 0, get)
    old = play_deal(config, 71, 1, get)
    config["candidates"].reverse()
    config["panels"].reverse()
    new = play_deal(config, 71, 1, factory())
    key = lambda row: (row["panel"], row["agent"], row["seat"])
    assert sorted(without_timing(old), key=key) == sorted(without_timing(new), key=key)


def test_paired_interval_clusters_seats_and_keeps_passes():
    first = [
        dict(
            deal_id=str(i),
            panel="fixed",
            seat=seat,
            deal_code=str(i),
            official_points=0,
        )
        for i in range(8)
        for seat in range(3)
    ]
    # Passes in one variant still belong in the comparison (0 points).
    second = [
        dict(r, official_points=12 if int(r["deal_id"]) % 2 else 0) for r in first
    ]
    estimate = paired_delta(first, second, "official_points", resamples=100)
    assert estimate["delta"] == 6
    assert estimate["independent_deals"] == 8
    assert estimate["rounds"] == 24
    low, high = estimate["interval_95"]
    assert low <= 6 <= high
    assert low > 0
    assert paired_delta(first, first, "official_points", resamples=100)[
        "interval_95"
    ] == [0, 0]


@pytest.mark.parametrize("change", ["missing", "duplicate", "deal", "unbalanced"])
def test_paired_comparison_rejects_invalid_inputs(change):
    first = [
        dict(
            deal_id=str(i),
            panel="fixed",
            seat=seat,
            deal_code=str(i),
            official_points=0,
        )
        for i in range(2)
        for seat in range(3)
    ]
    second = copy.deepcopy(first)
    if change == "missing":
        second.pop()
    elif change == "duplicate":
        second.append(second[0])
    elif change == "deal":
        second[0]["deal_code"] = "different"
    else:
        first.pop()
        second.pop()
    with pytest.raises(ValueError):
        paired_delta(first, second, "official_points", resamples=10)


@pytest.mark.parametrize("panel", ["all-neural", "mcts-default"])
def test_focused_mcts_report_has_common_references_and_separate_contrast(panel):
    config = dict(
        seeds=[71],
        deals_per_seed=2,
        candidates=["best", "all-neural", "mcts"],
        panels=[panel],
    )
    rows = []
    for i in range(2):
        for candidate, points in (("best", 10), ("all-neural", 20), ("mcts", 25)):
            for seat in range(3):
                rows.append(
                    dict(
                        deal_id=f"71:{i}",
                        deal_code=str(i),
                        run_seed=71,
                        agent=candidate,
                        panel=panel,
                        seat=seat,
                        seconds=0,
                        soloist=seat,
                        won=True,
                        official_points=points,
                        relative_points=points,
                        party_win=1,
                        seat_points=[points if s == seat else 0 for s in range(3)],
                        result=dict(
                            reason="won", declaration=dict(kind="grand", hand=False)
                        ),
                    )
                )
    result = report(config, rows)
    assert set(result["panels"]) == {panel}
    contrasts = result["panels"][panel]["differences_from_best"]
    assert contrasts["all-neural"]["official_points"]["delta"] == 10
    assert contrasts["all-neural-minus-mcts"]["official_points"]["delta"] == -5
    assert (
        contrasts["all-neural-minus-mcts"]["official_points"]["independent_deals"] == 2
    )
    assert "all-neural-minus-declaration" not in contrasts
    assert set(result["mixed_tables"]) == {
        f"{candidate}-vs-two-{panel}"
        for candidate in config["candidates"]
        if candidate != panel
    }
