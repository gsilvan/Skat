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

    def test_value(self) -> None:
        hand_value = (
            (Hand([Card(0, 0), Card(0, 1), Card(0, 2)]), 0),
            (Hand([Card(0, 5), Card(0, 6), Card(0, 7)]), 25),
            (Hand([Card(1, 3), Card(2, 0), Card(0, 7)]), 13),
        )
        for hand, value in hand_value:
            self.assertEqual(value, hand.value)

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

    def test_tgt(self) -> None:
        diamonds_order = Hand.Order(
            suits="♦♠♣♥", ranks="789QKXA", pivot_suits="♦♥♠♣", pivot_ranks="J"
        )

        self.assertTrue(Hand.tgt(Card(0, 1), Card(0, 0), diamonds_order))
        self.assertFalse(Hand.tgt(Card(0, 0), Card(0, 1), diamonds_order))

        self.assertTrue(Hand.tgt(Card(0, 6), Card(0, 0), diamonds_order))
        self.assertFalse(Hand.tgt(Card(0, 0), Card(0, 6), diamonds_order))

        self.assertTrue(Hand.tgt(Card(3, 3), Card(0, 3), diamonds_order))
        self.assertFalse(Hand.tgt(Card(0, 3), Card(3, 3), diamonds_order))

        for i in range(4):
            a = Card(i, 3)
            b = Card(0, 7)
            self.assertTrue(Hand.tgt(a, b, diamonds_order))
            self.assertFalse(Hand.tgt(b, a, diamonds_order))

        # Sheep head prep :)

        sheep_head = Hand.Order(
            suits="♦♥♣♠", ranks="789KXA", pivot_suits="♦♥♠♣", pivot_ranks="JQ"
        )

        self.assertTrue(Hand.tgt(Card(2, 0), Card(3, 0), sheep_head))
        self.assertFalse(Hand.tgt(Card(3, 0), Card(2, 0), sheep_head))

        for i in range(4):
            a = Card(i, 4)
            b = Card(i, 3)
            self.assertTrue(Hand.tgt(a, b, sheep_head))
            self.assertFalse(Hand.tgt(b, a, sheep_head))

    def test_eq(self) -> None:
        self.assertEqual(Hand([Card(0, 2), Card(1, 1)]), Hand([Card(0, 2), Card(1, 1)]))

    def test_sort(self) -> None:
        hearts_order = Hand.Order(
            suits="♦♠♣♥", ranks="789QKXA", pivot_suits="♦♥♠♣", pivot_ranks="J"
        )
        initial_hand = Hand(
            [Card(2, 2), Card(3, 3), Card(0, 3), Card(1, 2), Card(3, 4)]
        )
        initial_hand.sort(hearts_order)
        expected_hand = Hand(
            [Card(3, 3), Card(0, 3), Card(1, 2), Card(3, 4), Card(2, 2)]
        )
        self.assertEqual(expected_hand, initial_hand)
