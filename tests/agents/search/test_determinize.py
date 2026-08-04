import unittest
from random import Random

from skat.agents.search.determinize import SKAT_SLOT, deal_unseen, infer_voids
from skat.card import Card
from skat.games.suit import SuitGame, SuitGameTrick

CLUBS = 3
DIAMONDS = 0


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def trick(*turns) -> SuitGameTrick:
    played = SuitGameTrick(CLUBS)
    for player_id, c in turns:
        played.append(player_id, c)
    return played


class InferVoidsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prototype = SuitGameTrick(CLUBS)

    def test_no_voids_when_everybody_follows(self) -> None:
        played = trick((0, card("♦", "A")), (1, card("♦", "K")), (2, card("♦", "7")))
        voids = infer_voids([played], self.prototype)
        self.assertEqual({0: frozenset(), 1: frozenset(), 2: frozenset()}, voids)

    def test_discarding_marks_the_whole_suit_void(self) -> None:
        played = trick((0, card("♦", "A")), (1, card("♥", "7")), (2, card("♦", "K")))
        voids = infer_voids([played], self.prototype)
        self.assertEqual(set(SuitGame.suit_cards(DIAMONDS)), set(voids[1]))
        self.assertEqual(frozenset(), voids[0])
        self.assertEqual(frozenset(), voids[2])

    def test_jack_is_not_part_of_its_suit(self) -> None:
        # ♦J is trump in a club game, so being void in diamonds says nothing
        # about holding ♦J.
        played = trick((0, card("♦", "A")), (1, card("♥", "7")), (2, card("♦", "K")))
        voids = infer_voids([played], self.prototype)
        self.assertNotIn(card("♦", "J"), voids[1])

    def test_not_following_trump_marks_all_trumps_void(self) -> None:
        played = trick((0, card("♣", "J")), (1, card("♦", "7")), (2, card("♣", "A")))
        voids = infer_voids([played], self.prototype)
        self.assertEqual(set(SuitGame.trump_cards(CLUBS)), set(voids[1]))

    def test_voids_accumulate_over_tricks(self) -> None:
        first = trick((0, card("♦", "A")), (1, card("♥", "7")), (2, card("♦", "K")))
        second = trick((0, card("♠", "A")), (1, card("♥", "8")), (2, card("♠", "K")))
        voids = infer_voids([first, second], self.prototype)
        self.assertIn(card("♦", "7"), voids[1])
        self.assertIn(card("♠", "7"), voids[1])


class DealUnseenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = Random(7)

    def test_deals_the_requested_counts(self) -> None:
        unseen = list(SuitGame.suit_cards(DIAMONDS))  # 7 cards
        deal = deal_unseen(unseen, {1: 3, 2: 2, SKAT_SLOT: 2}, {}, self.rng)
        self.assertEqual(3, len(deal[1]))
        self.assertEqual(2, len(deal[2]))
        self.assertEqual(2, len(deal[SKAT_SLOT]))
        self.assertEqual(
            set(unseen), set(deal[1]) | set(deal[2]) | set(deal[SKAT_SLOT])
        )

    def test_respects_voids(self) -> None:
        diamonds = set(SuitGame.suit_cards(DIAMONDS))
        unseen = list(diamonds) + list(SuitGame.suit_cards(1))  # + hearts
        voids = {1: frozenset(diamonds)}
        for seed in range(20):
            deal = deal_unseen(unseen, {1: 5, 2: 5, SKAT_SLOT: 4}, voids, Random(seed))
            self.assertEqual(5, len(deal[1]))
            self.assertEqual(set(), diamonds & set(deal[1]))

    def test_skat_is_never_constrained(self) -> None:
        diamonds = set(SuitGame.suit_cards(DIAMONDS))
        unseen = list(diamonds) + list(SuitGame.suit_cards(1))
        voids = {1: frozenset(diamonds), 2: frozenset(diamonds)}
        deal = deal_unseen(unseen, {1: 3, 2: 4, SKAT_SLOT: 7}, voids, self.rng)
        # nobody else may hold a diamond, so all seven have to land in the skat
        self.assertEqual(diamonds, diamonds & set(deal[SKAT_SLOT]))

    def test_falls_back_when_constraints_are_unsatisfiable(self) -> None:
        unseen = list(SuitGame.suit_cards(DIAMONDS))
        # everyone is void in everything - impossible, but the search still
        # needs a world to run on
        voids = {1: frozenset(unseen), 2: frozenset(unseen)}
        deal = deal_unseen(unseen, {1: 3, 2: 4}, voids, self.rng)
        self.assertEqual(set(unseen), set(deal[1]) | set(deal[2]))

    def test_rejects_a_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            deal_unseen(SuitGame.suit_cards(DIAMONDS), {1: 3}, {}, self.rng)


if __name__ == "__main__":
    unittest.main()
