import unittest

import numpy as np

from skat.card import Card
from skat.games.suit import SuitGameTrick
from skat.trick import TrickHistory


class SuitGameTrickTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trick = SuitGameTrick(0)  # Diamonds

    def add_test_cards(self) -> None:
        player_cards = (
            (0, Card(3, 0)),  # value 0
            (1, Card(3, 5)),  # value 4
            (2, Card(3, 7)),  # value 11
        )
        for player, card in player_cards:
            self.trick.append(player_id=player, card=card)

    def test_empty_on_init(self) -> None:
        self.assertEqual(0, len(self.trick))

    def test_winner_empty(self) -> None:
        self.assertIsNone(self.trick.winner)

    def test_append_on_empty_trick(self) -> None:
        self.add_test_cards()
        self.assertEqual(3, len(self.trick))

    def test_append_too_many(self) -> None:
        self.add_test_cards()
        with self.assertRaises(Exception):
            self.trick.append(4, Card(0, 0))

    def test_value(self) -> None:
        self.add_test_cards()
        self.assertEqual(15, self.trick.value)

    def test_is_full(self) -> None:
        self.assertFalse(self.trick.is_full)
        self.add_test_cards()
        self.assertTrue(self.trick.is_full)

    def test_winner(self) -> None:
        self.add_test_cards()
        winning_player = self.trick.winner
        self.assertEqual(2, winning_player)

    def test_winner_with_empty_trick(self) -> None:
        self.assertIsNone(self.trick.winner)

    def test__is_trump_card(self) -> None:
        clubs_trick = SuitGameTrick(3)
        test_trump_cards = (
            Card(3, 0),
            Card(3, 7),
            Card(0, 3),  # This is a J
            Card(1, 3),
            Card(2, 3),
            Card(3, 3),
        )
        # compare with function
        for trump_card in test_trump_cards:
            self.assertTrue(clubs_trick.is_trump_card(trump_card))

    def test__is_trump_in_trick(self) -> None:
        # True test
        true_trick = SuitGameTrick(0)
        true_trick_cards = (Card(1, 1), Card(1, 1), Card(0, 6))
        for i, card in enumerate(true_trick_cards):
            true_trick.append(i + 1, card)
        self.assertTrue(true_trick.is_trump_in_trick())
        # False test
        false_trick = SuitGameTrick(1)
        false_trick_cards = (Card(2, 0), Card(2, 5), Card(2, 2))
        for i, card in enumerate(false_trick_cards):
            false_trick.append(i + 1, card)
        self.assertFalse(false_trick.is_trump_in_trick())
        # J test
        diamonds_trick = SuitGameTrick(3)
        diamonds_trick_cards = (
            Card(2, 4),
            Card(2, 3),  # Hearts J
            Card(2, 6),
        )
        for i, card in enumerate(diamonds_trick_cards):
            diamonds_trick.append(i + 1, card)
        self.assertTrue(diamonds_trick.is_trump_in_trick())

    def test_trick_winner(self) -> None:
        test_set = (
            # cards, trump_suit, winner
            ((Card(2, 1), Card(1, 3), Card(2, 5)), 0, 1),  # with j trump
            ((Card(2, 1), Card(2, 4), Card(3, 7)), 3, 2),  # with trump
            ((Card(0, 0), Card(0, 7), Card(0, 6)), 2, 1),  # without trump
            ((Card(0, 3), Card(1, 3), Card(2, 3)), 3, 2),  # j only
        )
        for t in test_set:
            trick = SuitGameTrick(t[1])
            for i, card in enumerate(t[0]):
                trick.append(i, card)
            self.assertEqual(t[2], trick.winner, msg=t)

    def test_is_trump(self) -> None:
        test_set = (
            ((Card(2, 0), Card(2, 1), Card(2, 2)), True),  # color trump
            ((Card(0, 3), Card(2, 1), Card(2, 4)), True),  # j trump
            ((Card(0, 4), Card(0, 5), Card(1, 3)), False),  # no trump in [0]
        )
        for test, is_trump in test_set:
            trick = SuitGameTrick(2)
            for idx, card in enumerate(test):
                trick.append(player_id=idx, card=card)
            self.assertIs(is_trump, trick.is_trump, msg=test)

    def test_forced_cards(self) -> None:
        trick = SuitGameTrick(0)
        self.assertEqual(set(), trick.forced_cards)
        trick.append(0, Card(1, 0))
        color_set = {
            Card(1, 0),
            Card(1, 1),
            Card(1, 2),
            Card(1, 4),
            Card(1, 5),
            Card(1, 6),
            Card(1, 7),
        }
        self.assertEqual(color_set, trick.forced_cards)

        trick = SuitGameTrick(0)  # renew trick
        trick.append(0, Card(0, 0))
        trump_set = {
            Card(0, 0),
            Card(0, 1),
            Card(0, 2),
            Card(0, 4),
            Card(0, 5),
            Card(0, 6),
            Card(0, 7),
            Card(0, 3),
            Card(1, 3),
            Card(2, 3),
            Card(3, 3),
        }
        self.assertEqual(trump_set, trick.forced_cards)

        trick = SuitGameTrick(0)  # renew trick
        trick.append(0, Card(3, 3))
        self.assertEqual(trump_set, trick.forced_cards)

    def test_as_vector(self) -> None:
        expected_vector = np.zeros(32)
        expected_vector[24], expected_vector[29], expected_vector[31] = 1, 1, 1
        self.add_test_cards()
        self.assertTrue(np.array_equal(expected_vector, self.trick.as_vector))


class TrickHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.history = TrickHistory()

    def add_tricks(self) -> None:
        first_trick = SuitGameTrick(0)
        first_trick.append(0, Card(1, 0))
        first_trick.append(1, Card(1, 1))
        first_trick.append(2, Card(1, 2))
        second_trick = SuitGameTrick(0)
        second_trick.append(0, Card(3, 0))
        second_trick.append(1, Card(3, 1))
        second_trick.append(2, Card(3, 2))
        self.history.append(first_trick)
        self.history.append(second_trick)

    def test_length(self) -> None:
        self.assertEqual(0, len(self.history))
        self.add_tricks()
        self.assertEqual(2, len(self.history))

    def test_to_numpy(self) -> None:
        exp_0 = np.zeros(32)
        self.assertTrue(np.array_equal(exp_0, self.history.to_numpy()))
        self.add_tricks()
        exp_1 = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
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
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.assertTrue(np.array_equal(exp_1, self.history.to_numpy()))

    def test_to_numpy_player_only(self) -> None:
        self.add_tricks()
        exp = np.array(
            [
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
                0,
                0,
            ]
        )
        self.assertTrue(np.array_equal(exp, self.history.to_numpy(player_id=0)))
