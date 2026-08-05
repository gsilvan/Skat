"""
What the rest of the codebase depends on is not the architecture, it is the
signature: `net(x)` for a `(B, 240)` float32 batch returns `(B, 32)` logits.
`PolicyAgent.select` and `distill.train` know nothing else about either
network, and that is the whole reason one can be swapped for the other by
checkpoint alone.

So this file pins the contract rather than the internals, plus the two
properties the experiment rests on: that the transformer is the smaller of the
two, and that its constant buffers stay out of the state dict.
"""

import unittest

import torch

from skat.models.card_transformer import CardTransformer
from skat.models.policy import PolicyNet


def parameters(net) -> int:
    return sum(p.numel() for p in net.parameters())


class ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.net = CardTransformer()
        self.net.eval()

    def test_it_is_shaped_like_the_observation_and_the_pack(self) -> None:
        self.assertEqual(PolicyNet.INPUT_SIZE, CardTransformer.INPUT_SIZE)
        self.assertEqual(PolicyNet.OUTPUT_SIZE, CardTransformer.OUTPUT_SIZE)

    def test_it_maps_a_batch_of_observations_to_a_batch_of_logits(self) -> None:
        out = self.net(torch.rand(3, CardTransformer.INPUT_SIZE))
        self.assertEqual((3, 32), tuple(out.shape))

    def test_a_batch_of_one_is_what_the_agent_actually_asks_for(self) -> None:
        # `PolicyAgent.select` passes `get_state_t`, which is (1, 240).
        out = self.net(torch.rand(1, CardTransformer.INPUT_SIZE))
        self.assertEqual((1, 32), tuple(out.shape))

    def test_it_tokenises_inside_the_forward(self) -> None:
        # The caller hands over a flat vector and nothing else. If this ever
        # needed a second argument, every call site would have to know which
        # architecture it was holding.
        self.assertEqual((1, 32), tuple(self.net(torch.zeros(1, 240)).shape))

    def test_a_narrow_observation_fails_at_the_forward_not_silently(self) -> None:
        with self.assertRaises(RuntimeError):
            self.net(torch.rand(1, CardTransformer.INPUT_SIZE - 1))

    def test_a_wide_observation_fails_too(self) -> None:
        # Every block is a fixed index, so a longer vector would slice cleanly
        # and be quietly wrong - the one case a shape error would not catch.
        with self.assertRaises(RuntimeError) as caught:
            self.net(torch.rand(1, CardTransformer.INPUT_SIZE + 1))
        self.assertIn("240", str(caught.exception))


class SizeTest(unittest.TestCase):
    def test_it_is_smaller_than_the_mlp_it_competes_with(self) -> None:
        # The premise of the whole exercise: the same corpus, a better prior,
        # fewer parameters. A refactor that inverts this has changed the
        # experiment, not just the code.
        self.assertLess(parameters(CardTransformer()), parameters(PolicyNet()))

    def test_the_constant_buffers_stay_out_of_the_state_dict(self) -> None:
        # They are rebuilt at construction. Storing them would mean adding or
        # renaming one retires every checkpoint written before it.
        stored = set(CardTransformer().state_dict())
        for buffer in ("status_index", "suit_of", "rank_of", "suit_onehot", "is_jack"):
            self.assertNotIn(buffer, stored)

    def test_its_dropout_is_real(self) -> None:
        torch.manual_seed(0)
        net = CardTransformer(dropout=0.5)
        x = torch.rand(8, CardTransformer.INPUT_SIZE)
        net.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), net(x)))
        net.train()
        with torch.no_grad():
            self.assertFalse(torch.equal(net(x), net(x)))


class FeatureSetTest(unittest.TestCase):
    """`features="raw"` is the control that separates "attention over a card
    set helped" from "the derived features helped"."""

    def test_the_derived_features_can_be_switched_off(self) -> None:
        net = CardTransformer(features="raw")
        net.eval()
        self.assertEqual((2, 32), tuple(net(torch.rand(2, 240)).shape))

    def test_the_raw_table_is_narrower(self) -> None:
        self.assertLess(
            CardTransformer.TOKEN_FEATURES["raw"],
            CardTransformer.TOKEN_FEATURES["full"],
        )

    def test_an_unknown_feature_set_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            CardTransformer(features="everything")
        self.assertIn("raw", str(caught.exception))

    def test_the_two_feature_sets_are_not_interchangeable_checkpoints(self) -> None:
        # The input width of `token` differs, so a silent load is impossible.
        with self.assertRaises(RuntimeError):
            CardTransformer(features="raw").load_state_dict(
                CardTransformer(features="full").state_dict()
            )


if __name__ == "__main__":
    unittest.main()
