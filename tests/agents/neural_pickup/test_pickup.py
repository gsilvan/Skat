from types import SimpleNamespace

import numpy as np
import pytest

from skat.__main__ import get_player
from skat.agents.neural_pickup.features import pickup_state
from skat.agents.neural_pickup.inference import NeuralPickup
from skat.hand import FULL_HAND


def view(cards=FULL_HAND[:10], bid=18):
    return SimpleNamespace(hand=cards, highest_bid=bid, seat_id=0, dealer=2)


class FixedScorer:
    def __init__(self, pickup, hand):
        self.values = pickup, hand

    def probabilities(self, state):
        assert state.shape == (36,)
        assert state[:32].sum() == 10
        return self.values


def test_features_only_use_original_hand_bid_and_position():
    cards = FULL_HAND[:10]
    first = pickup_state(cards, 18, 0, 2)
    assert first.shape == (36,)
    assert first[:32].sum() == 10
    assert first[32:35].sum() == 1
    assert not np.array_equal(first, pickup_state(cards, 24, 0, 2))
    assert not np.array_equal(first, pickup_state(cards, 18, 1, 2))
    with pytest.raises(ValueError):
        pickup_state((*cards[:9], cards[0]), 18, 0, 2)


def test_hand_requires_gain_and_safe_bid():
    hand = view()
    assert not NeuralPickup(FixedScorer(0.2, 0.8)).choose_pickup(hand)
    assert not NeuralPickup(FixedScorer(0.2, 0.8)).choose_pickup(view(bid=24))
    assert NeuralPickup(FixedScorer(0.5, 0.52), margin=0.03).choose_pickup(hand)
    assert NeuralPickup(FixedScorer(0.2, 0.8)).choose_pickup(view(bid=264))
    with pytest.raises(ValueError):
        NeuralPickup(FixedScorer(0.2, float("nan"))).choose_pickup(hand)


def test_pickup_strategy_is_independent_of_card_and_declaration_strategy():
    agent = get_player("random", pickup="neural", pickup_model=FixedScorer(0.2, 0.8))
    agent.set_state(view())
    assert agent.pickup_skat() is False
    assert agent.declaration_label == "heuristic"
    assert get_player("random").pickup_style == "always"
    with pytest.raises(ValueError, match="pickup-model"):
        get_player("random", pickup="neural")


def test_pickup_checkpoint_has_separate_schema(tmp_path):
    torch = pytest.importorskip("torch")
    from skat.agents.neural_pickup.checkpoint import load_pickup, save_pickup
    from skat.agents.neural_pickup.inference import PickupScorer
    from skat.models.pickup import PickupNet

    torch.set_num_threads(1)
    path = tmp_path / "pickup.pt"
    save_pickup(path, PickupNet(d_model=16, nhead=2, num_layers=1))
    scorer = PickupScorer(path)
    probabilities = scorer.probabilities(pickup_state(FULL_HAND[:10], 18, 0, 2))
    assert len(probabilities) == 2
    assert all(0 <= value <= 1 for value in probabilities)
    from skat.agents.neural_declaring.checkpoint import load_declaration

    with pytest.raises(ValueError, match="incompatible"):
        load_declaration(path)
    assert load_pickup(path)[1] == (1.0, 1.0)


def test_paired_rollouts_respect_the_original_bid(monkeypatch):
    from iss import pickup as importer
    from skat.agents.random import RandomAgent

    class ScriptedRandom(RandomAgent):
        choice = None

        def press_skat(self):
            return self.choice.pressed

        def declare_game(self, state=None):
            return self.choice.game

    monkeypatch.setattr(importer, "_agents", [ScriptedRandom(i) for i in range(3)])
    monkeypatch.setattr(importer, "_declarer", None)
    original = tuple(tuple(FULL_HAND[i : i + 10]) for i in (0, 10, 20)) + (
        tuple(FULL_HAND[30:]),
    )
    state, labels, historical_hand, historical_won = importer._simulate(
        ("test", original, 0, 400, False, True), 7
    )
    assert state.shape == (36,)
    assert labels == [0.0, 0.0]
    assert not historical_hand and historical_won
