"""
The two networks in this repo are the same shape and mean different things.

`SuitSoloNet` maps the 240-wide observation to 32 Q-values regressed against
`Round.cumulative_reward`; `PolicyNet` maps it to 32 policy logits trained
against MCTS visit counts. Nothing about the tensors distinguishes them, the
DQN checkpoints sit one flag away at `./trained_models/checkpoint_p*.pt`, and
loading one into the other produces an agent that plays confidently and wrongly
without a word of complaint. These tests are what makes that impossible.
"""

import os
import tempfile
import unittest

import torch

from skat.agents.distill.checkpoint import POLICY_FORMAT, load_policy, save_policy
from skat.models.policy import PolicyNet
from skat.models.suit_solo import SuitSoloNet

CPU = torch.device("cpu")


class RoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "policy.pt")

    def test_a_checkpoint_round_trips_to_the_same_predictions(self) -> None:
        torch.manual_seed(0)
        net = PolicyNet()
        net.eval()
        observation = torch.randn((1, PolicyNet.INPUT_SIZE))
        with torch.no_grad():
            before = net(observation)
        save_policy(self.path, net, meta={"samples": 17})
        restored, meta = load_policy(self.path, CPU)
        restored.eval()
        with torch.no_grad():
            after = restored(observation)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(17, meta["samples"])

    def test_it_carries_the_format_and_the_layout(self) -> None:
        save_policy(self.path, PolicyNet())
        blob = torch.load(self.path, weights_only=False)
        self.assertEqual(POLICY_FORMAT, blob["format"])
        self.assertEqual(PolicyNet.INPUT_SIZE, blob["input_size"])


class ConfusabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_a_dqn_checkpoint_is_refused(self) -> None:
        # Exactly what `DQN.optimize_model` writes: a bare state dict.
        path = os.path.join(self.directory.name, "checkpoint_p0.pt")
        net = SuitSoloNet()
        net(torch.zeros((1, SuitSoloNet.INPUT_SIZE)))  # materialize the lazy layers
        torch.save(net.state_dict(), path)
        with self.assertRaises(RuntimeError) as caught:
            load_policy(path, CPU)
        message = str(caught.exception)
        self.assertIn("DQN", message)
        self.assertIn("Q-values", message)
        self.assertIn("skat.agents.distill.train", message)

    def test_a_policy_checkpoint_is_refused_by_the_dqn(self) -> None:
        # The other direction. `DQN.__init__` hands whatever it loads straight
        # to `load_state_dict`, which chokes on the wrapper dict.
        path = os.path.join(self.directory.name, "policy.pt")
        save_policy(path, PolicyNet())
        with self.assertRaises(RuntimeError):
            SuitSoloNet().load_state_dict(torch.load(path, weights_only=False))

    def test_a_checkpoint_from_another_state_layout_is_refused(self) -> None:
        path = os.path.join(self.directory.name, "narrow.pt")
        save_policy(path, PolicyNet(input_size=64))
        with self.assertRaises(RuntimeError) as caught:
            load_policy(path, CPU)
        self.assertIn(str(PolicyNet.INPUT_SIZE), str(caught.exception))


class PolicyNetTest(unittest.TestCase):
    def test_it_is_shaped_like_the_observation_and_the_pack(self) -> None:
        self.assertEqual(SuitSoloNet.INPUT_SIZE, PolicyNet.INPUT_SIZE)
        self.assertEqual(32, PolicyNet.OUTPUT_SIZE)
        out = PolicyNet()(torch.zeros((3, PolicyNet.INPUT_SIZE)))
        self.assertEqual((3, 32), tuple(out.shape))

    def test_the_wrong_width_fails_at_the_forward_not_silently(self) -> None:
        with self.assertRaises(RuntimeError):
            PolicyNet()(torch.zeros((1, PolicyNet.INPUT_SIZE - 1)))

    def test_its_dropout_is_real(self) -> None:
        # `SuitSoloNet` takes a dropout argument and has no Dropout layer, so
        # train() and eval() are the same net. Here they must not be.
        torch.manual_seed(0)
        net = PolicyNet(dropout=0.5)
        observation = torch.randn((8, PolicyNet.INPUT_SIZE))
        net.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(net(observation), net(observation)))
        net.train()
        with torch.no_grad():
            self.assertFalse(torch.equal(net(observation), net(observation)))


if __name__ == "__main__":
    unittest.main()
