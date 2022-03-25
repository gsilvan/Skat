import unittest

import torch

from skat.agents.rl.mask import MaskedCategorical


class MaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor = torch.randn((1, 32))
        self.mask = torch.zeros((1, 32), dtype=bool)
        for i in [0, 2, 4, 8, 16]:
            self.mask[0][i] = True

    def test_masked_probs(self) -> None:
        m = MaskedCategorical(logits=self.tensor, mask=self.mask)
        self.assertNotAlmostEqual(0.0, float(m.probs[0][0]))
        self.assertAlmostEqual(0.0, float(m.probs[0][1]))
