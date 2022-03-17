import unittest

import numpy as np

from skat.card import Card
from skat.hand import Hand


class HandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.empty_hand = Hand()
        self.filled_hand = Hand([Card(0, 0), Card(3, 1), Card(3, 7)])

    def test_empty(self) -> None:
        self.assertEqual(0, len(self.empty_hand))

    def test_prefilled(self) -> None:
        self.assertEqual(3, len(self.filled_hand))

    def test_get(self) -> None:
        exp = Card(3, 1)
        self.assertEqual(exp, self.filled_hand[1])

    def test_set(self) -> None:
        old = Card(0, 0)
        new = Card(3, 3)
        self.assertEqual(old, self.filled_hand[0])
        self.filled_hand[0] = Card(3, 3)
        self.assertEqual(new, self.filled_hand[0])

    def test_del(self) -> None:
        del self.filled_hand[1]
        self.assertEqual(2, len(self.filled_hand))

    def test_str(self) -> None:
        pass

    def test_append(self) -> None:
        pass

    def test_insert(self) -> None:
        pass

    def test_as_vector(self) -> None:
        arr = np.array(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
            ]
        )
        self.assertTrue(np.array_equal(arr, self.filled_hand.as_vector))
