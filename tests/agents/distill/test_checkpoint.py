"""
Three networks in this repo take 240 floats and return 32 numbers, and all
three mean different things.

`QNet` maps the observation to Q-values regressed against
`Round.cumulative_reward`; `PolicyNet` maps it to policy logits trained against
MCTS visit counts; `CardTransformer` maps it to the same logits through a
different architecture. Nothing about the tensors distinguishes any of them,
the DQN checkpoints sit one flag away at `./trained_models/checkpoint_p*.pt`,
and loading the wrong one produces an agent that plays confidently and wrongly
without a word of complaint. These tests are what makes that impossible.

The other property here is back-compatibility. The `arch` key arrived with the
second architecture, so every checkpoint older than it - including the one
holding the measured MLP baseline - has to keep loading.
"""

import os
import tempfile
import unittest

import torch

from skat.agents.distill.checkpoint import POLICY_FORMAT, load_policy, save_policy
from skat.models import ARCHITECTURES
from skat.models.card_transformer import CardTransformer
from skat.models.policy import PolicyNet
from skat.models.q_net import QNet

CPU = torch.device("cpu")
BASELINE = "trained_models/policy.pt"


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
        torch.save(QNet().state_dict(), path)
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
            QNet().load_state_dict(torch.load(path, weights_only=False))

    def test_a_checkpoint_from_another_state_layout_is_refused(self) -> None:
        path = os.path.join(self.directory.name, "narrow.pt")
        save_policy(path, PolicyNet(input_size=64))
        with self.assertRaises(RuntimeError) as caught:
            load_policy(path, CPU)
        self.assertIn(str(PolicyNet.INPUT_SIZE), str(caught.exception))


class ArchitectureDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        torch.manual_seed(0)

    def path(self, name: str) -> str:
        return os.path.join(self.directory.name, name)

    def test_a_transformer_checkpoint_comes_back_as_a_transformer(self) -> None:
        path = self.path("transformer.pt")
        save_policy(path, CardTransformer())
        net, _ = load_policy(path, CPU)
        self.assertIsInstance(net, CardTransformer)

    def test_a_transformer_round_trips_to_the_same_predictions(self) -> None:
        path = self.path("transformer.pt")
        net = CardTransformer()
        net.eval()
        observation = torch.rand((1, CardTransformer.INPUT_SIZE))
        with torch.no_grad():
            before = net(observation)
        save_policy(path, net)
        restored, _ = load_policy(path, CPU)
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(before, restored(observation)))

    def test_an_arch_less_checkpoint_is_still_an_mlp(self) -> None:
        # Built by hand, because `save_policy` always writes the key now. This
        # is the shape of every checkpoint from before the second architecture
        # existed, and it has to keep loading.
        path = self.path("legacy.pt")
        net = PolicyNet()
        torch.save(
            {
                "format": POLICY_FORMAT,
                "input_size": net.INPUT_SIZE,
                "output_size": net.OUTPUT_SIZE,
                "state_dict": net.state_dict(),
                "meta": {},
            },
            path,
        )
        restored, _ = load_policy(path, CPU)
        self.assertIsInstance(restored, PolicyNet)

    @unittest.skipUnless(os.path.exists(BASELINE), f"no {BASELINE} in this tree")
    def test_the_shipped_baseline_checkpoint_still_loads(self) -> None:
        # Weak - it is skipped wherever the artefact is absent - but it is the
        # only test that touches the real file rather than a fixture.
        net, _ = load_policy(BASELINE, CPU)
        self.assertIsInstance(net, PolicyNet)

    def test_an_unknown_architecture_names_the_ones_it_knows(self) -> None:
        path = self.path("future.pt")
        save_policy(path, PolicyNet(), arch="perceiver")
        with self.assertRaises(RuntimeError) as caught:
            load_policy(path, CPU)
        message = str(caught.exception)
        self.assertIn("perceiver", message)
        for known in ARCHITECTURES:
            self.assertIn(known, message)

    def test_the_arch_kwargs_survive_the_round_trip(self) -> None:
        path = self.path("raw.pt")
        save_policy(
            path, CardTransformer(features="raw"), arch_kwargs={"features": "raw"}
        )
        net, _ = load_policy(path, CPU)
        self.assertEqual("raw", net.features)

    def test_a_transformer_checkpoint_is_refused_by_the_mlp(self) -> None:
        path = self.path("transformer.pt")
        save_policy(path, CardTransformer())
        blob = torch.load(path, weights_only=False)
        with self.assertRaises(RuntimeError):
            PolicyNet().load_state_dict(blob["state_dict"])

    def test_an_mlp_checkpoint_is_refused_by_the_transformer(self) -> None:
        path = self.path("mlp.pt")
        save_policy(path, PolicyNet())
        blob = torch.load(path, weights_only=False)
        with self.assertRaises(RuntimeError):
            CardTransformer().load_state_dict(blob["state_dict"])

    def test_an_unregistered_network_cannot_be_saved_unnamed(self) -> None:
        # Saving one would produce a file nothing can load.
        with self.assertRaises(ValueError) as caught:
            save_policy(self.path("stray.pt"), QNet())
        self.assertIn("ARCHITECTURES", str(caught.exception))


class PolicyNetTest(unittest.TestCase):
    def test_it_is_shaped_like_the_observation_and_the_pack(self) -> None:
        self.assertEqual(QNet.INPUT_SIZE, PolicyNet.INPUT_SIZE)
        self.assertEqual(32, PolicyNet.OUTPUT_SIZE)
        out = PolicyNet()(torch.zeros((3, PolicyNet.INPUT_SIZE)))
        self.assertEqual((3, 32), tuple(out.shape))

    def test_the_wrong_width_fails_at_the_forward_not_silently(self) -> None:
        with self.assertRaises(RuntimeError):
            PolicyNet()(torch.zeros((1, PolicyNet.INPUT_SIZE - 1)))

    def test_its_dropout_is_real(self) -> None:
        # `QNet` takes a dropout argument and has no Dropout layer, so
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
