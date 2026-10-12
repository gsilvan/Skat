import json
from types import SimpleNamespace

import numpy as np
import pytest

from skat.agents.bidding import BiddingStyle, playable
from skat.agents.neural_bidding.features import bidding_state
from skat.agents.random import RandomAgent
from skat.game import Round
from skat.hand import FULL_HAND


def test_features_use_original_hand_and_relative_position():
    cards = FULL_HAND[:10]
    state = bidding_state(cards, 0, 2)
    assert state[:32].sum() == 10 and state[32:].tolist() == [1, 0, 0]
    np.testing.assert_array_equal(state, bidding_state(cards, 1, 0))
    with pytest.raises(ValueError):
        bidding_state(cards[:9], 0, 2)
    with pytest.raises(ValueError):
        bidding_state([cards[0]] * 10, 0, 2)


class StubBidder:
    threshold = 0.7

    def __init__(self, p):
        self.p, self.calls = p, 0

    def probability(self, cards, seat, dealer):
        self.calls += 1
        assert len(cards) == 10
        return self.p


def test_neural_gate_replaces_heuristic_and_caches_once_per_round():
    cards = FULL_HAND[0:3] + FULL_HAND[8:11] + FULL_HAND[16:19] + [FULL_HAND[24]]
    assert not playable(set(cards))
    agent = RandomAgent().configure_bidding("neural", StubBidder(0.8))
    agent.set_state(SimpleNamespace(hand=cards, seat_id=0, dealer=2))
    assert agent.max_bid() >= 18
    assert agent.bid(0) > 0
    assert agent.neural_bidder.calls == 1
    agent.new_round_event()
    agent.max_bid()
    assert agent.neural_bidder.calls == 2
    agent.configure_bidding("conservative")
    assert agent.max_bid() == 0
    agent.configure_bidding("neural", StubBidder(0.1))
    assert agent.max_bid() == 0
    with pytest.raises(ValueError):
        agent.configure_bidding("neural")


def test_accepted_hands_keep_safe_ceiling_and_covering_declaration():
    for seed in range(10):
        agents = [
            RandomAgent(i).configure_bidding("neural", StubBidder(1)) for i in range(3)
        ]
        r = Round(seed=seed, agents=agents, start=False)
        r.start()
        assert r.result is not None and not r.result.overbid


def test_checkpoint_train_and_mix_with_all_card_agents(tmp_path):
    torch = pytest.importorskip("torch")
    from skat.__main__ import get_player, main
    from skat.agents.distill.checkpoint import save_policy
    from skat.agents.neural_bidding.checkpoint import load_bidding, save_bidding
    from skat.agents.neural_bidding.inference import NeuralBidder
    from skat.agents.neural_bidding.samples import BiddingCorpus, BiddingWriter
    from skat.agents.neural_bidding.train import calibrate, probability_metrics, train
    from skat.config import MCTSConfig, TrainConfig
    from skat.models.bidding import BiddingNet
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    writer = BiddingWriter(corpus, shard_rows=3)
    for split in ("train", "val", "calibration", "test"):
        for i in range(8):
            writer.add(
                bidding_state(FULL_HAND[i : i + 10], i % 3, 2),
                i % 2 == 0,
                split,
                {"group": f"{split}-{i}"},
            )
    writer.finish({"test": True})
    config = TrainConfig(epochs=1, batch_size=8, d_model=16, nhead=2, num_layers=1)
    path = tmp_path / "bid.pt"
    report = train(corpus, path, config=config)
    assert report["test"]["rows"] == 8 and report["best_epoch"] == 1
    assert report["test"]["brier"] >= 0
    model, temp, meta = load_bidding(path)
    assert meta["test"] == report["test"]
    x = torch.from_numpy(bidding_state(FULL_HAND[:10], 0, 2)).unsqueeze(0)
    assert torch.equal(model(x), model(x)) and temp > 0
    bidder = NeuralBidder(path, threshold=0)
    policy = tmp_path / "policy.pt"
    save_policy(
        policy, CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    )
    agents = [
        get_player(
            name,
            seed=i,
            policy=policy,
            mcts=MCTSConfig(12, 2),
            bidding="neural",
            bidder=bidder,
        )
        for i, name in enumerate(("random", "mcts", "policy"))
    ]
    r = Round(agents=agents, seed=7, start=False)
    r.start()
    assert not r.result.overbid
    main(
        [
            "play",
            "--agents",
            "random",
            "random",
            "random",
            "--bidding0",
            "neural",
            "--bidding-model",
            str(path),
            "--rounds",
            "1",
            "--workers",
            "1",
            "--bidding-threshold",
            "0",
        ]
    )
    assert BiddingStyle.NEURAL == agents[2].bidding
    with pytest.raises(ValueError):
        get_player("mcts", bidding="neural")
    assert calibrate([10, 10, -10, -10], [1, 0, 0, 1]) > 1
    assert probability_metrics([0, 0], [0, 1])["brier"] == 0.25
    saved = torch.load(path, weights_only=True)
    saved["feature_schema"] = "old"
    torch.save(saved, path)
    with pytest.raises(ValueError):
        load_bidding(path)
    # Card policies must never load as hand classifiers.
    with pytest.raises(ValueError):
        load_bidding(policy)
    save_bidding(path, BiddingNet(d_model=16, nhead=2, num_layers=1))
    shard = next(corpus.glob("train-*/states.npy"))
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        BiddingCorpus(corpus)
    assert (
        json.loads(path.with_suffix(".training.json").read_text())["completed_epochs"]
        == 1
    )


def test_bidding_evaluation_only_changes_candidate_and_counts_passes(tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.distill.checkpoint import save_policy
    from skat.agents.neural_bidding.checkpoint import save_bidding
    from skat.evaluation import evaluate, paired_points_comparison
    from skat.models.bidding import BiddingNet
    from skat.models.card_transformer import CardTransformer

    torch.set_num_threads(1)
    bidder = tmp_path / "bid.pt"
    save_bidding(bidder, BiddingNet(d_model=16, nhead=2, num_layers=1))
    policy = tmp_path / "policy.pt"
    save_policy(
        policy, CardTransformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    )
    a = evaluate(["random"], policy, tmp_path / "a.json", deals=2, seed=7)
    b = evaluate(
        ["random"],
        policy,
        tmp_path / "b.json",
        deals=2,
        seed=7,
        bidding="neural",
        bidding_model=bidder,
        bidding_threshold=1,
    )
    assert b["manifest"]["bidding"]["reference_style"] == "conservative"
    assert all(r["bidding_ceiling"] == 0 for r in b["outcomes"])
    assert len(b["outcomes"]) == 12
    points = paired_points_comparison(a["outcomes"], b["outcomes"])
    assert points["paired"] == 12
    assert b["summary"]["random"]["points_per_deal"] == 0
    first = [dict(panel="p", seat=0, seed=1, won=True, official_points=-40)]
    second = [dict(panel="p", seat=0, seed=1, won=None, official_points=0)]
    assert paired_points_comparison(first, second)["delta"] == 40
    with pytest.raises(ValueError):
        paired_points_comparison(first, [])
    with pytest.raises(ValueError):
        paired_points_comparison(first + first, second)


def test_threshold_selection_uses_points_then_frozen_independent_test(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from contextlib import nullcontext

    import skat.agents.neural_bidding.benchmark as module

    calls = []

    def fake(names, policy, output, **kwargs):
        threshold = (
            kwargs["bidding_threshold"] if kwargs["bidding"] == "neural" else None
        )
        calls.append((kwargs["seed"], threshold))
        score = 10 if threshold in (0.70, 0.75) else 0
        summary = {"points_per_deal": score, "win_rate": 0 if score else 1}
        rows = [
            dict(panel=panel, seat=seat, seed=i, official_points=score, won=True)
            for panel in ("random", "policy")
            for seat in range(3)
            for i in range(kwargs["deals"])
        ]
        return {
            "summary": {names[0]: summary},
            "outcomes": rows,
            "panels": {panel: {names[0]: summary} for panel in ("random", "policy")},
        }

    monkeypatch.setattr(module, "evaluate", fake)
    monkeypatch.setattr(module, "SearchPool", lambda *a, **k: nullcontext(None))
    report = module.benchmark_bidding(
        tmp_path / "benchmark",
        agent="random",
        policy="card",
        reference_policy="reference",
        bidding_model="bidding",
        validation_deals=2,
        deals=3,
    )
    assert report["selected_threshold"] == 0.70
    assert calls[-2:] == [(43, None), (43, 0.70)]
    assert all(seed == 42 for seed, threshold in calls[:-2])
    assert report["test_points_paired"]["paired"] == 18
    assert report["test_points_paired"]["delta"] == 10
    assert report["state"] == "complete"
