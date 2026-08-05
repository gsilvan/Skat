"""
Collecting the seed data and sizing a gradient step are two separate decisions.

They used to be one: `optimize_model` gated on `len(buffer) < batch_size`, and
`batch_size` was 10000. That made the batch 40x a normal DQN's just to buy a
warm-up, and it meant the warm-up itself was mostly spent acting greedily on a
freshly initialised network - one arbitrary deterministic policy, replayed for
~950 of the ~1000 rounds.
"""

import unittest
from unittest.mock import patch

import torch

from skat.agents.rl import dqn as dqn_module
from skat.agents.rl.buffer import ReplayBuffer


def bare_dqn(learn_start: int = 10, batch_size: int = 4, **kwargs):
    """A DQN with no nets, no writer and no io - just the schedule fields."""
    dqn = dqn_module.DQN.__new__(dqn_module.DQN)
    dqn.replay_buffer = ReplayBuffer(buffer_size=1000)
    dqn.learn_start = learn_start
    dqn.batch_size = batch_size
    dqn.epsilon = 0.98
    dqn.epsilon_decay = 0.5  # coarse, so one step is unmistakable
    dqn.epsilon_min = 0.10
    dqn.episode = 0
    dqn.learning = False
    dqn.device = torch.device("cpu")
    for name, value in kwargs.items():
        setattr(dqn, name, value)
    return dqn


def fill(dqn, count: int) -> None:
    for _ in range(count):
        dqn.replay_buffer.push(
            state=torch.zeros((1, 4)),
            action=torch.zeros((1, 32), dtype=torch.long),
            next_state=torch.zeros((1, 4)),
            next_mask=torch.ones(32, dtype=torch.bool),
            reward=torch.zeros((1, 1)),
        )


class WarmUpGateTest(unittest.TestCase):
    def test_no_gradient_step_below_learn_start(self) -> None:
        dqn = bare_dqn(learn_start=10)
        fill(dqn, 9)
        with patch.object(dqn.replay_buffer, "sample") as sample:
            dqn.optimize_model()
        sample.assert_not_called()

    def test_the_step_happens_at_learn_start(self) -> None:
        dqn = bare_dqn(learn_start=10, batch_size=4)
        fill(dqn, 10)
        with (
            patch.object(
                dqn.replay_buffer, "sample", side_effect=RuntimeError("sampled")
            ),
            self.assertRaisesRegex(RuntimeError, "sampled"),
        ):
            dqn.optimize_model()

    def test_the_batch_is_batch_size_not_learn_start(self) -> None:
        # the whole point of the split: a 10000-transition warm-up must not
        # imply a 10000-sample gradient step
        dqn = bare_dqn(learn_start=10, batch_size=4)
        fill(dqn, 10)
        with patch.object(
            dqn.replay_buffer, "sample", side_effect=RuntimeError("stop")
        ) as sample:
            self.assertRaises(RuntimeError, dqn.optimize_model)
        self.assertEqual(4, sample.call_args.kwargs["batch_size"])

    def test_the_episode_counter_still_runs_during_warm_up(self) -> None:
        # epsilon and loss are logged against it; a counter that only starts
        # after the warm-up would misalign every scalar in tensorboard
        dqn = bare_dqn(learn_start=10)
        for _ in range(3):
            dqn.optimize_model()
        self.assertEqual(3, dqn.episode)


class ExplorationRateTest(unittest.TestCase):
    def setUp(self) -> None:
        # keep every explore=True call on the random branch, so these tests
        # never need a network behind them
        always_explore = patch.object(dqn_module.random, "random", return_value=0.0)
        always_explore.start()
        self.addCleanup(always_explore.stop)

    def test_it_is_flat_one_during_warm_up(self) -> None:
        dqn = bare_dqn(learn_start=10)
        fill(dqn, 9)
        self.assertTrue(dqn.warming_up)
        self.assertEqual(1.0, dqn.exploration_rate)

    def test_epsilon_is_untouched_during_warm_up(self) -> None:
        dqn = bare_dqn(learn_start=10)
        fill(dqn, 5)
        for _ in range(20):
            dqn.select_action(torch.zeros((1, 4)), self.mask(), explore=True)
        self.assertEqual(0.98, dqn.epsilon)

    def test_it_falls_back_to_epsilon_afterwards(self) -> None:
        dqn = bare_dqn(learn_start=10)
        fill(dqn, 10)
        self.assertFalse(dqn.warming_up)
        self.assertEqual(0.98, dqn.exploration_rate)

    def test_it_decays_once_warm_up_is_over(self) -> None:
        dqn = bare_dqn(learn_start=10)
        fill(dqn, 10)
        dqn.select_action(torch.zeros((1, 4)), self.mask(), explore=True)
        self.assertAlmostEqual(0.49, dqn.epsilon)

    def test_it_does_not_decay_on_an_evaluation_pass(self) -> None:
        # epsilon is never read when explore=False, so decaying there just
        # burns the schedule of whatever run comes next
        dqn = bare_dqn(learn_start=0, policy_net=lambda state: torch.zeros((1, 32)))
        with patch.object(dqn_module, "MaskedCategorical") as masked:
            masked.return_value.probs.argmax.return_value = torch.tensor(0)
            dqn.select_action(torch.zeros((1, 4)), self.mask(), explore=False)
        self.assertEqual(0.98, dqn.epsilon)

    def test_it_never_falls_below_the_floor(self) -> None:
        dqn = bare_dqn(learn_start=0)
        for _ in range(50):
            dqn.select_action(torch.zeros((1, 4)), self.mask(), explore=True)
        self.assertEqual(dqn.epsilon_min, dqn.epsilon)

    @staticmethod
    def mask() -> torch.Tensor:
        mask = torch.zeros(32, dtype=torch.bool)
        mask[0] = True
        return mask


class ConstructionTest(unittest.TestCase):
    def test_buffer_size_is_honoured(self) -> None:
        # it was accepted as an argument and then thrown away: the buffer was
        # built with a hardcoded 100000 whatever the caller asked for
        dqn = dqn_module.DQN(device="cpu", buffer_size=5, learn_start=5, batch_size=5)
        self.assertEqual(5, dqn.replay_buffer.buffer.maxlen)

    def test_the_default_capacity_is_unchanged(self) -> None:
        dqn = dqn_module.DQN(device="cpu")
        self.assertEqual(100000, dqn.replay_buffer.buffer.maxlen)

    def test_learn_start_below_batch_size_is_refused(self) -> None:
        # random.sample would otherwise raise a bare "Sample larger than
        # population" on the first gradient step, thousands of rounds in
        with self.assertRaises(ValueError) as caught:
            dqn_module.DQN(device="cpu", learn_start=100, batch_size=256)
        message = str(caught.exception)
        self.assertIn("100", message)
        self.assertIn("256", message)

    def test_the_defaults_are_consistent(self) -> None:
        dqn = dqn_module.DQN(device="cpu")
        self.assertGreaterEqual(dqn.learn_start, dqn.batch_size)


if __name__ == "__main__":
    unittest.main()
