import unittest

from skat.card import Card
from skat.utils.misc import disjoint


class UtilsTest(unittest.TestCase):
    def test_disjoint(self):
        simple_list = [
            (
                1,
                2,
                3,
            ),
            (4, 5),
            (42, 23),
        ]
        self.assertTrue(disjoint(simple_list))
        simple_list.append((3, 0))
        self.assertFalse(disjoint(simple_list))
        card_list = [
            (
                Card(0, 0),
                Card(3, 5),
                Card(2, 1),
            ),
            (
                Card(1, 0),
                Card(1, 5),
                Card(1, 1),
            ),
            (
                Card(2, 0),
                Card(2, 5),
                Card(2, 2),
            ),
        ]
        self.assertTrue(disjoint(card_list))
        card_list.append((Card(1, 5), Card(1, 2)))
        self.assertFalse(disjoint(card_list))
