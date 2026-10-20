import copy

import pytest

pytest.importorskip("torch")

from scripts.rate_agents import PROFILES, merge_ratings  # noqa: E402


def document(profile, source):
    return dict(
        protocol="play-seat0-random-conservative-v1",
        rounds=1000,
        seed=42,
        candidate_seat=0,
        dealer_rotation=True,
        reference_agents=["random", "random"],
        reference_bidding="conservative",
        source_commit=source,
        source_sha256=source,
        measured_at=source,
        device="cpu",
        torch_version="test",
        profiles=[dict(id=profile, policy_sha256=source)],
        status="complete",
    )


def test_partial_refresh_keeps_old_hashes_and_per_profile_provenance():
    old, new = document("random", "old"), document("all-neural", "new")
    snapshot = copy.deepcopy(old)
    merged = merge_ratings(old, new)
    assert old == snapshot
    assert [r["id"] for r in merged["profiles"]] == ["random", "all-neural"]
    assert merged["profiles"][0]["policy_sha256"] == "old"
    assert merged["profiles"][0]["measurement"]["source_commit"] == "old"
    assert merged["profiles"][1]["measurement"]["source_commit"] == "new"
    refreshed = merge_ratings(merged, document("random", "latest"))
    assert refreshed["profiles"][0]["policy_sha256"] == "latest"
    assert refreshed["profiles"][1]["measurement"]["source_commit"] == "new"


@pytest.mark.parametrize(
    "key,value",
    [
        ("rounds", 2000),
        ("seed", 43),
        ("candidate_seat", 1),
        ("reference_bidding", "neural"),
    ],
)
def test_partial_refresh_rejects_different_protocols(key, value):
    old, new = document("random", "old"), document("all-neural", "new")
    new[key] = value
    with pytest.raises(ValueError, match="different rating protocols"):
        merge_ratings(old, new)


def test_fixed_random_baseline_and_neural_combinations_are_retained():
    profiles = {p[0]: p for p in PROFILES}
    assert profiles["random"][2:6] == ("random", None, 600, "conservative")
    assert profiles["policy-iss-all-neural"][5] == "neural"
    assert profiles["policy-iss-all-neural"][6] == "neural"
    assert profiles["policy-iss-all-neural"][8] == "neural"
    assert profiles["mcts-600-all-neural"][2] == "mcts"
    assert profiles["mcts-600-all-neural"][5] == "neural"
    assert profiles["mcts-600-all-neural"][6] == "neural"
    assert profiles["mcts-600-all-neural"][8] == "neural"
