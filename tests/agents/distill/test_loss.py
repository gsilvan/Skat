"""
The loss must not see the cards the agent cannot play.

A policy over 32 card slots is asked about a hand of at most ten, and in the
middle of a trick usually about three or four. Everything else is not a bad
move, it is not a move at all - it must not contribute to the loss and it must
not receive a gradient, or the network spends its capacity learning to rank
cards it will never be offered.

The other thing pinned here is that the masking is out-of-place.
`skat.agents.rl.mask.MaskedCategorical` does the same job with a
`masked_fill_`, which is fine for a no-grad argmax and wrong in a backward
pass.
"""

import unittest

import torch

from skat.agents.distill.loss import masked_log_softmax, policy_loss


def mask_of(*slots: int) -> torch.Tensor:
    mask = torch.zeros((1, 32), dtype=torch.bool)
    for slot in slots:
        mask[0][slot] = True
    return mask


def target_over(*slots: int) -> torch.Tensor:
    target = torch.zeros((1, 32))
    for slot in slots:
        target[0][slot] = 1.0 / len(slots)
    return target


class IllegalCardsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.mask = mask_of(3, 7, 11)
        self.target = target_over(3, 7, 11)
        self.logits = torch.randn((1, 32))

    def loss_with(self, logits: torch.Tensor) -> torch.Tensor:
        return policy_loss(logits, self.target, self.mask)

    def test_an_illegal_logit_cannot_change_the_loss(self) -> None:
        before = self.loss_with(self.logits)
        for shove in (1e6, -1e6):
            moved = self.logits.clone()
            moved[0][20] = shove  # 20 is not in the mask
            self.assertEqual(before.item(), self.loss_with(moved).item())

    def test_the_gradient_of_an_illegal_logit_is_exactly_zero(self) -> None:
        logits = self.logits.clone().requires_grad_(True)
        self.loss_with(logits).backward()
        assert logits.grad is not None
        self.assertEqual(0.0, logits.grad[0][~self.mask[0]].abs().sum().item())
        # ...and the legal ones do get one, or the test above proves nothing.
        self.assertGreater(logits.grad[0][self.mask[0]].abs().sum().item(), 0.0)

    def test_a_zero_target_never_produces_a_nan(self) -> None:
        # The illegal slots carry `finfo.min` after masking; multiplied by a
        # target of 0 they must drop out rather than turn the sum into a nan.
        loss = self.loss_with(self.logits)
        self.assertTrue(torch.isfinite(loss))

    def test_it_is_out_of_place(self) -> None:
        # MaskedCategorical would have overwritten these.
        before = self.logits.clone()
        masked_log_softmax(self.logits, self.mask)
        self.assertTrue(torch.equal(before, self.logits))


class CrossEntropyTest(unittest.TestCase):
    def test_a_perfect_prediction_costs_the_target_entropy(self) -> None:
        mask = mask_of(0, 1, 2, 3)
        target = torch.zeros((1, 32))
        target[0][0], target[0][1], target[0][2], target[0][3] = 0.5, 0.25, 0.2, 0.05
        # Logits whose masked softmax is exactly the target.
        logits = torch.full((1, 32), -50.0)
        logits[0][:4] = torch.log(target[0][:4])
        entropy = -(target[0][:4] * torch.log(target[0][:4])).sum()
        self.assertAlmostEqual(
            entropy.item(), policy_loss(logits, target, mask).item(), places=4
        )

    def test_a_confident_wrong_answer_costs_more_than_a_hedged_one(self) -> None:
        mask = mask_of(0, 1)
        target = torch.zeros((1, 32))
        target[0][0] = 1.0
        hedged = torch.zeros((1, 32))
        confident = torch.zeros((1, 32))
        confident[0][1] = 5.0
        self.assertGreater(
            policy_loss(confident, target, mask).item(),
            policy_loss(hedged, target, mask).item(),
        )

    def test_the_batch_is_averaged_not_summed(self) -> None:
        mask = mask_of(0, 1).repeat(4, 1)
        target = target_over(0, 1).repeat(4, 1)
        logits = torch.zeros((4, 32))
        one = policy_loss(logits[:1], target[:1], mask[:1])
        four = policy_loss(logits, target, mask)
        self.assertAlmostEqual(one.item(), four.item(), places=6)


class MaskedLogSoftmaxTest(unittest.TestCase):
    def test_the_legal_probabilities_sum_to_one(self) -> None:
        mask = mask_of(5, 9)
        probs = masked_log_softmax(torch.randn((1, 32)), mask).exp()
        self.assertAlmostEqual(1.0, probs[0][mask[0]].sum().item(), places=5)
        self.assertAlmostEqual(0.0, probs[0][~mask[0]].sum().item(), places=5)

    def test_rows_are_masked_independently(self) -> None:
        logits = torch.zeros((2, 32))
        masks = torch.cat([mask_of(0), mask_of(1, 2, 3)])
        probs = masked_log_softmax(logits, masks).exp()
        self.assertAlmostEqual(1.0, probs[0][0].item(), places=5)
        self.assertAlmostEqual(0.0, probs[1][0].item(), places=5)


if __name__ == "__main__":
    unittest.main()
