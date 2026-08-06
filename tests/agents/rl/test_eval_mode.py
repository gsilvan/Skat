"""
The DQN never used to put a network into eval mode, or turn autograd off.

With `QNet` that was only latent - it has no dropout and no batch norm, so
train mode and eval mode compute the same function, and the wasted autograd
graph was merely wasted. It stops being latent the moment `--dqn-arch
transformer` is used, because `CardTransformer` has dropout: a training-mode
forward would make `select_action` stochastic on top of the epsilon-greedy
schedule, including on evaluation passes where epsilon is never read, and
would turn the bootstrap target into a random variable.

So these cases run against a network that *does* have dropout, because against
`QNet` every one of them would pass whether or not the fix is there.
"""

import unittest

import torch
from torch import nn

from skat.agents.rl.dqn import DQN_ARCHITECTURES, evaluating
from skat.models.observation import INPUT_SIZE


class Dropping(nn.Module):
    """A minimal net whose output depends on the mode, and loudly."""

    INPUT_SIZE = INPUT_SIZE
    OUTPUT_SIZE = 32

    def __init__(self, **_) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_SIZE, 32)
        self.dropout = nn.Dropout(0.9)

    def forward(self, x):
        return self.dropout(self.linear(x))


class EvaluatingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.net = Dropping()

    def test_it_turns_the_net_deterministic_inside_the_block(self) -> None:
        x = torch.rand((4, INPUT_SIZE))
        with evaluating(self.net):
            self.assertTrue(torch.equal(self.net(x), self.net(x)))

    def test_it_turns_autograd_off_inside_the_block(self) -> None:
        with evaluating(self.net):
            self.assertFalse(self.net(torch.rand((1, INPUT_SIZE))).requires_grad)

    def test_it_puts_a_training_net_back_into_training_mode(self) -> None:
        self.net.train()
        with evaluating(self.net):
            self.assertFalse(self.net.training)
        self.assertTrue(self.net.training)

    def test_it_leaves_an_eval_net_in_eval_mode(self) -> None:
        # Restores what was there, rather than assuming train().
        self.net.eval()
        with evaluating(self.net):
            pass
        self.assertFalse(self.net.training)

    def test_it_restores_the_mode_even_when_the_body_raises(self) -> None:
        self.net.train()
        with self.assertRaises(KeyboardInterrupt), evaluating(self.net):
            raise KeyboardInterrupt  # a Ctrl-C mid-round is the realistic case
        self.assertTrue(self.net.training)


def bare_dqn(**kwargs):
    """A DQN with the two nets and nothing else - no writer, no buffer, no io."""
    from skat.agents.rl import dqn as dqn_module
    from skat.agents.rl.buffer import ReplayBuffer

    dqn = dqn_module.DQN.__new__(dqn_module.DQN)
    dqn.device = torch.device("cpu")
    dqn.policy_net = Dropping()
    dqn.target_net = Dropping()
    dqn.replay_buffer = ReplayBuffer(buffer_size=10)
    dqn.epsilon = 0.0
    dqn.epsilon_decay = 1.0
    dqn.epsilon_min = 0.0
    dqn.learn_start = 0
    for name, value in kwargs.items():
        setattr(dqn, name, value)
    return dqn


class SelectActionTest(unittest.TestCase):
    """Action selection must be stochastic only through epsilon."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.dqn = bare_dqn()
        self.state = torch.rand((1, INPUT_SIZE))
        self.mask = torch.zeros(32, dtype=torch.bool)
        self.mask[:5] = True

    def choices(self, n: int, explore: bool) -> set:
        return {
            int(self.dqn.select_action(self.state, self.mask, explore=explore).argmax())
            for _ in range(n)
        }

    def test_a_greedy_choice_is_the_same_card_every_time(self) -> None:
        # With dropout live this would wander across the legal cards.
        self.assertEqual(1, len(self.choices(20, explore=False)))

    def test_an_evaluation_pass_is_deterministic(self) -> None:
        # epsilon is never read when explore=False, so nothing else may be
        # random either.
        self.assertEqual(1, len(self.choices(20, explore=False)))

    def test_it_leaves_the_policy_net_in_training_mode(self) -> None:
        # The very next thing that happens to it is a gradient step.
        self.dqn.policy_net.train()
        self.dqn.select_action(self.state, self.mask, explore=False)
        self.assertTrue(self.dqn.policy_net.training)

    def test_choosing_a_card_builds_no_autograd_graph(self) -> None:
        before = torch.is_grad_enabled()
        action = self.dqn.select_action(self.state, self.mask, explore=False)
        self.assertFalse(action.requires_grad)
        self.assertEqual(before, torch.is_grad_enabled())


class BootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.dqn = bare_dqn()
        self.states = torch.rand((6, INPUT_SIZE))
        self.masks = torch.zeros((6, 32), dtype=torch.bool)
        self.masks[:, :4] = True

    def test_the_target_is_not_a_random_variable(self) -> None:
        # The headline: TD learning chases a moving target already, and
        # dropout on the target net would add multiplicative noise to it.
        first = self.dqn.bootstrap(self.states, self.masks)
        second = self.dqn.bootstrap(self.states, self.masks)
        self.assertTrue(torch.equal(first, second))

    def test_the_target_carries_no_gradient(self) -> None:
        self.assertFalse(self.dqn.bootstrap(self.states, self.masks).requires_grad)

    def test_it_leaves_the_target_net_as_it_found_it(self) -> None:
        self.dqn.target_net.train()
        self.dqn.bootstrap(self.states, self.masks)
        self.assertTrue(self.dqn.target_net.training)


class ArchitectureTest(unittest.TestCase):
    def test_the_registry_names_a_class_and_its_kwargs(self) -> None:
        for name, (model, kwargs) in DQN_ARCHITECTURES.items():
            net = model(**kwargs)
            self.assertEqual(
                (2, 32), tuple(net(torch.rand((2, INPUT_SIZE))).shape), name
            )

    def test_the_transformer_is_built_without_dropout(self) -> None:
        # Otherwise the comparison against the dense net would confound the
        # architecture with a regularisation change.
        model, kwargs = DQN_ARCHITECTURES["transformer"]
        net = model(**kwargs)
        net.train()
        x = torch.rand((4, INPUT_SIZE))
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), net(x)))

    def test_the_dense_default_is_the_q_net_not_the_policy_net(self) -> None:
        # `PolicyNet` is the same geometry but carries dropout for supervised
        # training, which is the wrong net to regress Q-values with.
        from skat.models import PolicyNet, QNet

        model, _ = DQN_ARCHITECTURES["mlp"]
        self.assertIs(QNet, model)
        self.assertIsNot(PolicyNet, model)


if __name__ == "__main__":
    unittest.main()
