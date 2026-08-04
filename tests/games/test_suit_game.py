import unittest

from skat.card import Card
from skat.games.suit import SuitGame


class SuitGameTest(unittest.TestCase):
    def test_find_all_trump_cards(self) -> None:
        for i in range(4):
            suit_game = SuitGame(i)
            trumps_test_set = (
                Card(i, 0),  # ♦7
                Card(i, 1),  # ♦8
                Card(i, 2),  # ♦9
                Card(i, 4),  # ♦Q
                Card(i, 5),  # ♦K
                Card(i, 6),  # ♦10
                Card(i, 7),  # ♦A
                Card(0, 3),  # ♦J
                Card(1, 3),  # ♥J
                Card(2, 3),  # ♠J
                Card(3, 3),  # ♣J
            )
            computed_set = suit_game.trump_cards(i)
            self.assertEqual(trumps_test_set, computed_set)

    def test_find_all_suit_cards(self) -> None:
        for i in range(4):
            suit_test_set = (
                Card(i, 0),  # ♦7
                Card(i, 1),  # ♦8
                Card(i, 2),  # ♦9
                Card(i, 4),  # ♦Q
                Card(i, 5),  # ♦K
                Card(i, 6),  # ♦10
                Card(i, 7),  # ♦A
            )
            computed_set = SuitGame.suit_cards(i)
            self.assertEqual(suit_test_set, computed_set)

    def test_game_values(self) -> None:
        test_set = (9, 10, 11, 12)
        for i in range(4):
            suit_game = SuitGame(i)
            self.assertEqual(test_set[i], suit_game.value)

    def test_new_trick(self) -> None:
        suit_game = SuitGame(0)
        hash_old = hash(suit_game.trick)
        suit_game.new_trick()
        hash_new = hash(suit_game.trick)
        self.assertNotEqual(hash_old, hash_new)

    def test_is_trump_card(self) -> None:
        trick = SuitGame(3).trick  # clubs
        trumps = (
            Card(3, 0),  # ♣7  - the trump suit itself
            Card(3, 7),  # ♣A
            Card(0, 3),  # ♦J  - every jack is trump, whatever its suit
            Card(1, 3),  # ♥J
            Card(2, 3),  # ♠J
            Card(3, 3),  # ♣J
        )
        self.assertTrue(frozenset(trumps) <= frozenset(SuitGame.trump_cards(3)))
        for card in trumps:
            self.assertTrue(trick.is_trump_card(card), card)

    def test_is_not_trump_card(self) -> None:
        trick = SuitGame(1).trick  # hearts
        for card in (Card(0, 5), Card(0, 1), Card(2, 4), Card(3, 1)):
            self.assertFalse(trick.is_trump_card(card), card)

    def test_trump_and_suit_tables_are_precomputed(self) -> None:
        """
        The tables are built once at import, not per call - rebuilding them was
        the hottest thing in the MCTS search. Returning the identical object is
        the contract that keeps it that way; they are tuples, so sharing them
        is safe.
        """
        for suit in range(4):
            self.assertIs(SuitGame.trump_cards(suit), SuitGame.trump_cards(suit))
            self.assertIs(SuitGame.suit_cards(suit), SuitGame.suit_cards(suit))

    def test_bug_8(self):
        # trick=[Turn(player_id=1, card=♣A), Turn(player_id=2, card=♣7), Turn(player_id=0, card=♠A)]
        trump_suit = 2
        game = SuitGame(trump_suit)
        player_cards = (
            (1, Card(3, 7)),
            (2, Card(3, 0)),
            (0, Card(trump_suit, 7)),
        )
        for player, card in player_cards:
            game.trick.append(player_id=player, card=card)
        self.assertEqual(0, game.trick.winner)

    def test_bug_9(self):
        # trick=[Turn(player_id=0, card=♥J), Turn(player_id=1, card=♠Q), Turn(player_id=2, card=♦J)]
        trump_suit = 2
        game = SuitGame(trump_suit)
        player_cards = (
            (0, Card(1, 3)),
            (1, Card(2, 4)),
            (2, Card(0, 3)),
        )
        for player, card in player_cards:
            game.trick.append(player_id=player, card=card)
        self.assertEqual(0, game.trick.winner)

    def test_bug_10(self):
        # trick=[Turn(player_id=0, card=♥J), Turn(player_id=1, card=♠K), Turn(player_id=2, card=♣X)]
        trump_suit = 2
        game = SuitGame(trump_suit)
        player_cards = (
            (0, Card(1, 3)),
            (1, Card(2, 5)),
            (2, Card(3, 6)),
        )
        for player, card in player_cards:
            game.trick.append(player_id=player, card=card)
        self.assertEqual(0, game.trick.winner)

    def test_not_a_bug_11(self) -> None:
        # trick = [Turn(player_id=1, card=♦8), Turn(player_id=2, card=♦K), Turn(player_id=0, card=♣J)]
        trump_suit = 2
        game = SuitGame(trump_suit)
        player_cards = (
            (1, Card(0, 1)),
            (2, Card(0, 5)),
            (0, Card(3, 3)),
        )
        for player, card in player_cards:
            game.trick.append(player_id=player, card=card)
        self.assertEqual(0, game.trick.winner)
