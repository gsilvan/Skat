import json

import pytest

from skat.compare import compare


def evaluations(tmp_path):
    manifest = dict(
        rules_schema="rules",
        observation_schema="state",
        source_sha256="source",
        seed=42,
        deals=2,
        deal_offset=0,
        opponent_policy_sha256="fixed",
        mcts={"simulations": 600},
        mcts_rng_scheme="worlds",
        agents=["policy"],
    )
    rows = [
        dict(
            agent="policy",
            panel=panel,
            seat=seat,
            seed=deal,
            soloist=0,
            won=(seat == 0) if deal else None,
            seconds=1,
            official_points=1,
            result={"declaration": {"kind": "grand"} if deal else None},
        )
        for panel in ("random", "policy")
        for seat in range(3)
        for deal in range(2)
    ]
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text(json.dumps(dict(manifest=manifest, outcomes=rows)))
    return paths


def test_compare_handles_passed_games_and_fixed_references(tmp_path):
    a, b = evaluations(tmp_path)
    result = compare(a, b, tmp_path / "comparison.json")
    assert result["comparisons"]["policy"]["random"]["paired"]["delta"] == 0
    assert result["candidate_by_game"]["policy"]["grand"]["played"] == 3


@pytest.mark.parametrize("change", ["reference", "incomplete", "duplicate", "source"])
def test_compare_rejects_unfair_or_partial_comparisons(tmp_path, change):
    a, b = evaluations(tmp_path)
    document = json.loads(b.read_text())
    if change == "reference":
        document["manifest"]["opponent_policy_sha256"] = "other"
    elif change == "source":
        document["manifest"]["source_sha256"] = "other"
    elif change == "incomplete":
        document["outcomes"].pop()
    else:
        document["outcomes"][-1] = document["outcomes"][0]
    b.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        compare(a, b)
