"""
The training loop, end to end and tiny.

The property worth guarding here is the train/validation split. The ~22
decisions of one round share a hand, a soloist and a trick history, so a
row-wise split puts near-duplicates on both sides: the validation curve then
tracks the training curve, overfitting looks like convergence, and the one
number that says whether to collect more data says the opposite.
"""

import os
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

from skat.agents.distill.checkpoint import load_policy
from skat.agents.distill.loss import policy_loss
from skat.agents.distill.samples import ShardWriter, load_shards
from skat.agents.distill.train import (
    agreement,
    batches,
    build_targets,
    split_by_group,
    train,
)
from skat.agents.policy import PolicyAgent
from skat.hand import FULL_HAND
from skat.models.policy import PolicyNet


def corpus(directory: str, rounds: int = 20, per_round: int = 8) -> None:
    """
    A corpus a network can actually fit: every position with the same three
    legal cards, and the same one always most visited.
    """
    writer = ShardWriter(directory, input_size=PolicyNet.INPUT_SIZE, tag="test")
    moves = [FULL_HAND[i] for i in (0, 1, 2)]
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        writer.new_round()
        for _ in range(per_round):
            state = rng.random(PolicyNet.INPUT_SIZE).astype(np.float32)
            writer.add(
                state=state,
                totals={moves[0]: [80, 72.0], moves[1]: [15, 7.5], moves[2]: [5, 1.0]},
                moves=moves,
                choice=moves[0],
            )
    writer.close()


def arguments(data: str, out: str, **overrides) -> Namespace:
    args = Namespace(
        data=data,
        out=out,
        epochs=8,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-5,
        dropout=0.0,
        target="visits",
        target_temperature=1.0,
        val_fraction=0.2,
        seed=0,
        device="cpu",
        log_dir=os.path.join(os.path.dirname(out), "runs"),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class SplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = torch.tensor([g for g in range(10) for _ in range(5)])

    def test_the_split_never_puts_one_round_on_both_sides(self) -> None:
        train_idx, val_idx = split_by_group(self.groups, 0.2, seed=0)
        self.assertEqual(
            set(),
            set(self.groups[train_idx].tolist()) & set(self.groups[val_idx].tolist()),
        )

    def test_it_covers_every_row_exactly_once(self) -> None:
        train_idx, val_idx = split_by_group(self.groups, 0.2, seed=0)
        self.assertEqual(
            sorted(train_idx.tolist() + val_idx.tolist()), list(range(len(self.groups)))
        )

    def test_it_holds_out_roughly_the_asked_for_fraction(self) -> None:
        _, val_idx = split_by_group(self.groups, 0.2, seed=0)
        self.assertEqual(10, len(val_idx))  # two of ten rounds, five rows each

    def test_a_tiny_corpus_still_gets_a_validation_set(self) -> None:
        _, val_idx = split_by_group(torch.tensor([0, 0, 1, 1]), 0.01, seed=0)
        self.assertGreater(len(val_idx), 0)

    def test_the_seed_decides_which_rounds_are_held_out(self) -> None:
        first = split_by_group(self.groups, 0.3, seed=0)[1].tolist()
        self.assertEqual(first, split_by_group(self.groups, 0.3, seed=0)[1].tolist())
        self.assertNotEqual(first, split_by_group(self.groups, 0.3, seed=5)[1].tolist())


class BatchTest(unittest.TestCase):
    def test_every_row_appears_once_per_pass(self) -> None:
        generator = torch.Generator().manual_seed(0)
        index = torch.arange(10)
        seen = [int(i) for batch in batches(index, 3, generator) for i in batch]
        self.assertEqual(list(range(10)), sorted(seen))

    def test_the_last_batch_is_short_rather_than_dropped(self) -> None:
        generator = torch.Generator().manual_seed(0)
        sizes = [len(b) for b in batches(torch.arange(10), 3, generator)]
        self.assertEqual([3, 3, 3, 1], sizes)


class AgreementTest(unittest.TestCase):
    def test_it_scores_the_best_legal_card_not_the_best_card(self) -> None:
        logits = torch.zeros((1, 32))
        logits[0][9] = 5.0  # illegal favourite
        logits[0][1] = 1.0
        masks = torch.zeros((1, 32), dtype=torch.bool)
        masks[0][0], masks[0][1] = True, True
        self.assertEqual(1.0, agreement(logits, masks, torch.tensor([1])))
        self.assertEqual(0.0, agreement(logits, masks, torch.tensor([0])))


class TrainingRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = os.path.join(self.directory.name, "shards")
        self.out = os.path.join(self.directory.name, "policy.pt")
        corpus(self.data)

    def test_a_few_epochs_on_a_memorizable_set_drive_the_loss_down(self) -> None:
        samples = load_shards(self.data, input_size=PolicyNet.INPUT_SIZE)
        targets = build_targets(samples, "visits", 1.0)
        torch.manual_seed(0)
        before = PolicyNet(dropout=0.0)
        before.eval()
        with torch.no_grad():
            start = policy_loss(before(samples.states), targets, samples.masks).item()

        train(arguments(self.data, self.out, epochs=15))

        net, _ = load_policy(self.out, torch.device("cpu"), dropout=0.0)
        net.eval()
        with torch.no_grad():
            end = policy_loss(net(samples.states), targets, samples.masks).item()
        self.assertLess(end, start)

    def test_it_writes_a_checkpoint_a_policy_agent_can_load(self) -> None:
        train(arguments(self.data, self.out, epochs=1))
        self.assertTrue(os.path.exists(self.out))
        agent = PolicyAgent(path=self.out, device="cpu")
        self.assertEqual(20, agent.meta["rounds"])
        self.assertEqual(160, agent.meta["samples"])

    def test_the_value_target_trains_too(self) -> None:
        train(arguments(self.data, self.out, target="value", target_temperature=0.05))
        self.assertTrue(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
