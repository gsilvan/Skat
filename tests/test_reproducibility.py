import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from skat.agents.random import RandomAgent
from skat.agents.rl.buffer import ReplayBuffer
from skat.agents.rl.dqn import DQN
from skat.game import Tournament


def tournament_snapshot(global_seed: int):
    random.seed(global_seed)
    tournament = Tournament(
        rounds=12,
        agents=[RandomAgent(seed=100 + seat) for seat in range(3)],
        seed=99,
    )
    tournament.start()
    return (
        tuple(tournament.scores),
        tournament.passed,
        tuple(
            (
                score.solo_rounds,
                score.solo_wins,
                score.defence_rounds,
                score.defence_wins,
                score.card_points,
            )
            for score in tournament.seat_scores
        ),
    )


class RandomReproducibilityTest(unittest.TestCase):
    def test_agent_streams_do_not_depend_on_global_random(self):
        self.assertEqual(tournament_snapshot(1), tournament_snapshot(987654321))

    def test_replay_sampling_has_a_private_seeded_stream(self):
        def sampled(global_seed):
            random.seed(global_seed)
            buffer = ReplayBuffer(buffer_size=20, seed=42)
            for value in range(20):
                buffer.push(
                    state=value,
                    action=value,
                    next_state=None,
                    next_mask=None,
                    reward=value,
                )
            return [row.state for row in buffer.sample(8)]

        self.assertEqual(sampled(1), sampled(2))


class TorchReproducibilityTest(unittest.TestCase):
    def test_dqn_seed_controls_initial_weights_without_leaking_global_state(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("skat.agents.rl.dqn.SummaryWriter"),
        ):
            first = DQN(
                device="cpu",
                seed=17,
                model_path=str(Path(directory) / "first.pt"),
            )
            torch.rand(100)
            second = DQN(
                device="cpu",
                seed=17,
                model_path=str(Path(directory) / "second.pt"),
            )
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(
                        first.policy_net.state_dict().values(),
                        second.policy_net.state_dict().values(),
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
