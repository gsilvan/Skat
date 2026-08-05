"""
Soft targets, and what temperature does to them.

The teacher's opinion arrives as raw visit counts and summed values, one row
per decision, 32 wide with only the legal cards filled in. Turning that into a
distribution is where the two ways of reading a search - how often it looked at
a move, and how well the move scored - become two different training targets.
Both must be zero outside the mask and both must normalize, or the loss is
training on cards that were never on the table.
"""

import unittest

import torch

from skat.agents.distill.targets import value_targets, visit_targets


def rows(*counts: list[float]) -> torch.Tensor:
    """A (len(counts), 32) tensor whose first entries are `counts`."""
    out = torch.zeros((len(counts), 32))
    for row, values in enumerate(counts):
        out[row][: len(values)] = torch.tensor(values)
    return out


def mask_of(width: int, *slots: int) -> torch.Tensor:
    mask = torch.zeros((width, 32), dtype=torch.bool)
    for row in range(width):
        for slot in slots:
            mask[row][slot] = True
    return mask


class VisitTargetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.visits = rows([60.0, 30.0, 10.0])
        self.masks = mask_of(1, 0, 1, 2)

    def test_the_distribution_sums_to_one_over_the_legal_cards(self) -> None:
        target = visit_targets(self.visits, self.masks)
        self.assertAlmostEqual(1.0, target.sum().item(), places=5)

    def test_it_is_zero_outside_the_mask(self) -> None:
        # A count that landed on an illegal slot - which should never happen,
        # but the target must not carry it if it does.
        visits = self.visits.clone()
        visits[0][9] = 500.0
        target = visit_targets(visits, self.masks)
        self.assertEqual(0.0, target[0][~self.masks[0]].sum().item())
        self.assertAlmostEqual(1.0, target.sum().item(), places=5)

    def test_temperature_one_is_the_plain_normalized_counts(self) -> None:
        target = visit_targets(self.visits, self.masks, temperature=1.0)
        self.assertAlmostEqual(0.6, target[0][0].item(), places=5)
        self.assertAlmostEqual(0.3, target[0][1].item(), places=5)
        self.assertAlmostEqual(0.1, target[0][2].item(), places=5)

    def test_a_low_temperature_sharpens_towards_the_most_visited_move(self) -> None:
        plain = visit_targets(self.visits, self.masks, temperature=1.0)
        sharp = visit_targets(self.visits, self.masks, temperature=0.25)
        self.assertGreater(sharp[0][0].item(), plain[0][0].item())
        self.assertLess(sharp[0][2].item(), plain[0][2].item())
        self.assertAlmostEqual(1.0, sharp.sum().item(), places=5)

    def test_a_high_temperature_flattens_it(self) -> None:
        flat = visit_targets(self.visits, self.masks, temperature=100.0)
        self.assertAlmostEqual(1 / 3, flat[0][0].item(), places=2)

    def test_rows_normalize_independently(self) -> None:
        visits = rows([60.0, 30.0, 10.0], [1.0, 1.0, 2.0])
        target = visit_targets(visits, mask_of(2, 0, 1, 2))
        self.assertAlmostEqual(1.0, target[0].sum().item(), places=5)
        self.assertAlmostEqual(1.0, target[1].sum().item(), places=5)
        self.assertAlmostEqual(0.5, target[1][2].item(), places=5)


class ValueTargetsTest(unittest.TestCase):
    def setUp(self) -> None:
        # The much-visited move scores worse per visit than the rare one - the
        # case where the two targets genuinely disagree.
        self.visits = rows([60.0, 30.0, 10.0])
        self.values = rows([30.0, 21.0, 9.0])  # means 0.50, 0.70, 0.90
        self.masks = mask_of(1, 0, 1, 2)

    def test_it_sums_to_one_and_is_zero_outside_the_mask(self) -> None:
        target = value_targets(self.values, self.visits, self.masks)
        self.assertAlmostEqual(1.0, target.sum().item(), places=5)
        self.assertEqual(0.0, target[0][~self.masks[0]].sum().item())

    def test_it_ranks_moves_the_way_the_teacher_did(self) -> None:
        # `MCTSAgent.best` maximises value_sum / visits, so the target must put
        # its mass on slot 2 even though slot 0 was visited six times as often.
        target = value_targets(self.values, self.visits, self.masks)
        self.assertEqual(2, int(target[0].argmax()))
        self.assertGreater(target[0][2].item(), target[0][1].item())
        self.assertGreater(target[0][1].item(), target[0][0].item())

    def test_an_unvisited_legal_move_gets_no_mass(self) -> None:
        # `MCTSAgent.best` scores an unvisited move -1.0; entering it with a
        # value of 0 would make it look average instead.
        visits = self.visits.clone()
        values = self.values.clone()
        visits[0][3], values[0][3] = 0.0, 0.0
        masks = mask_of(1, 0, 1, 2, 3)
        target = value_targets(values, visits, masks)
        self.assertEqual(0.0, target[0][3].item())
        self.assertAlmostEqual(1.0, target.sum().item(), places=5)

    def test_a_low_temperature_sharpens_towards_the_best_mean(self) -> None:
        soft = value_targets(self.values, self.visits, self.masks, temperature=1.0)
        sharp = value_targets(self.values, self.visits, self.masks, temperature=0.05)
        self.assertGreater(sharp[0][2].item(), soft[0][2].item())


if __name__ == "__main__":
    unittest.main()
