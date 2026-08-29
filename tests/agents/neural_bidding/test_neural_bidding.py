import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from skat.agents.bidding import BiddingStyle, ceiling, playable
from skat.agents.mcts import MCTSAgent
from skat.agents.neural_bidding.checkpoint import load_bidding, save_bidding
from skat.agents.neural_bidding.collect import collect
from skat.agents.neural_bidding.features import bidding_state, relative_position
from skat.agents.neural_bidding.samples import (
    BiddingShardWriter,
    load_bidding_shards,
)
from skat.agents.neural_bidding.train import train
from skat.agents.random import RandomAgent
from skat.game import Round
from skat.hand import FULL_HAND
from skat.models.bidding import BiddingNet


class FeatureTest(unittest.TestCase):
    def test_encodes_cards_and_each_relative_position(self):
        cards = FULL_HAND[:10]
        for seat, dealer, expected in ((1, 0, 0), (2, 0, 1), (0, 0, 2)):
            state = bidding_state(cards, seat, dealer)
            self.assertEqual((35,), state.shape)
            np.testing.assert_array_equal(state[:32], [1] * 10 + [0] * 22)
            np.testing.assert_array_equal(
                state[32:], np.eye(3, dtype=np.float32)[expected]
            )
            self.assertEqual(expected, relative_position(seat, dealer))

    def test_requires_the_original_ten_cards(self):
        with self.assertRaisesRegex(ValueError, "ten-card"):
            bidding_state(FULL_HAND[:9], 0, 0)


class FakeBidder:
    threshold = 0.5

    def __init__(self, probability):
        self.value = probability

    def probability(self, cards, seat, dealer):
        return self.value


class AgentGateTest(unittest.TestCase):
    def round_with(self, agent, first_hand=None):
        first_hand = list(first_hand or FULL_HAND[:10])
        remaining = [card for card in FULL_HAND if card not in first_hand]
        deal = [first_hand, remaining[:10], remaining[10:20], remaining[20:]]
        round_ = Round(
            agents=[agent, RandomAgent(), RandomAgent()],
            initial_cards=deal,
            start=False,
        )
        round_.deal()
        return round_

    def test_negative_prediction_passes(self):
        agent = RandomAgent(bidding=BiddingStyle.NEURAL, seed=0)
        agent.set_neural_bidder(FakeBidder(0.49))
        self.round_with(agent)
        self.assertEqual(0, agent.max_bid())
        self.assertEqual(0.49, agent.last_bidding_probability)

    def test_positive_prediction_uses_safe_ceiling(self):
        agent = RandomAgent(bidding=BiddingStyle.NEURAL, seed=0)
        agent.set_neural_bidder(FakeBidder(0.5))
        round_ = self.round_with(agent)
        expected = ceiling(
            BiddingStyle.CONSERVATIVE,
            agent.intended_game(),
            set(round_.player[0].hand),
        )
        self.assertEqual(expected, agent.max_bid())

    def test_mcts_neural_gate_replaces_playable_heuristic(self):
        agent = MCTSAgent(bidding=BiddingStyle.NEURAL, seed=0)
        agent.set_neural_bidder(FakeBidder(1.0))
        weak = [FULL_HAND[index] for index in (0, 1, 2, 8, 9, 10, 16, 17, 24, 25)]
        self.assertFalse(playable(set(weak)))
        round_ = self.round_with(agent, weak)
        expected = ceiling(
            BiddingStyle.CONSERVATIVE,
            agent.intended_game(),
            set(round_.player[0].hand),
        )
        self.assertEqual(expected, agent.max_bid())


class DataAndCheckpointTest(unittest.TestCase):
    def test_rewards_are_signed_and_groups_survive_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BiddingShardWriter(directory, shard_size=2, tag="test")
            writer.add(bidding_state(FULL_HAND[:10], 0, 0), True, group=0)
            writer.add(bidding_state(FULL_HAND[10:20], 1, 0), False, group=0)
            writer.add(bidding_state(FULL_HAND[20:30], 2, 0), True, group=1)
            writer.close()
            samples = load_bidding_shards(directory)
        self.assertEqual([1.0, -1.0, 1.0], samples.rewards.tolist())
        self.assertEqual([1.0, 0.0, 1.0], samples.labels.tolist())
        self.assertEqual(samples.groups[0], samples.groups[1])
        self.assertNotEqual(samples.groups[1], samples.groups[2])

    def test_checkpoint_round_trip_and_rejects_another_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bid.pt"
            net = BiddingNet(hidden_size=32, dropout=0.0)
            save_bidding(str(path), net, {"source": "test"})
            restored, meta = load_bidding(str(path), torch.device("cpu"))
            self.assertEqual("test", meta["source"])
            for left, right in zip(net.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))
            torch.save({"format": "skat-policy-v1"}, path)
            with self.assertRaisesRegex(RuntimeError, "not a bidding checkpoint"):
                load_bidding(str(path), torch.device("cpu"))

    def test_tiny_collection_and_training_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            states = []
            rewards = []
            for run in range(2):
                data = root / f"data-{run}"
                stats = collect(
                    str(data), deals=2, simulations=1, determinizations=1, seed=7
                )
                self.assertEqual(6, stats["samples"])
                samples = load_bidding_shards(str(data))
                states.append(samples.states)
                rewards.append(samples.rewards)
                checkpoint = root / f"model-{run}.pt"
                train(
                    argparse.Namespace(
                        data=str(data),
                        out=str(checkpoint),
                        epochs=1,
                        batch_size=3,
                        lr=3e-4,
                        weight_decay=0.01,
                        dropout=0.0,
                        hidden_size=32,
                        val_fraction=0.5,
                        threshold=0.5,
                        log_dir=str(root / f"logs-{run}"),
                        seed=11,
                        device="cpu",
                    )
                )
                checkpoints.append(
                    torch.load(checkpoint, map_location="cpu", weights_only=False)
                )
            self.assertTrue(torch.equal(states[0], states[1]))
            self.assertTrue(torch.equal(rewards[0], rewards[1]))
            for name, value in checkpoints[0]["state_dict"].items():
                self.assertTrue(torch.equal(value, checkpoints[1]["state_dict"][name]))


if __name__ == "__main__":
    unittest.main()
