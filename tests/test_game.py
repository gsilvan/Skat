import unittest

from skat.card import Card
from skat.game import GamePhase, Round
from skat.games.suit import SuitGame


class GameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.round = Round(
            dealer=0,
            skip_bidding=True,
            solo_player_id=0,
            declare_game=SuitGame(suit=3),
            start=False,
            seed=42,
        )

    def test_front_hand(self) -> None:
        self.assertEqual(1, self.round.front_hand)

    def test_middle_hand(self) -> None:
        self.assertEqual(2, self.round.middle_hand)

    def test_back_hand(self) -> None:
        self.assertEqual(0, self.round.back_hand)

    def test_points_soloist(self) -> None:
        self.assertEqual(0, self.round.points_soloist)
        # TODO: check points after some cards were played

    def test_points_defenders(self) -> None:
        self.assertEqual(0, self.round.points_defenders)
        # TODO: check points after some cards were played

    def test_deal_from_deck(self) -> None:
        self.assertEqual(0, len(self.round._deck))
        self.round.deal()
        for player in self.round._player:
            self.assertEqual(10, len(player.hand))

    def test_deal_from_provided_cards(self) -> None:
        cards = (
            (Card(0, 0), Card(0, 1), Card(0, 2)),
            (Card(1, 0), Card(1, 1), Card(1, 2)),
            (Card(2, 0), Card(2, 1), Card(2, 2)),
            (Card(3, 0), Card(3, 1)),
        )
        custom_round = Round(
            dealer=0,
            skip_bidding=True,
            solo_player_id=0,
            declare_game=SuitGame(suit=3),
            start=False,
            initial_cards=cards,
        )
        for player in custom_round._player:
            self.assertEqual(0, len(player.hand))
        self.assertEqual(0, len(custom_round._skat))
        custom_round.deal()
        for player in custom_round._player:
            self.assertEqual(3, len(player.hand))
        self.assertEqual(2, len(custom_round._skat))

    def test_next_player(self) -> None:
        self.round.deal()
        self.round._phase = GamePhase.PLAYING
        # next_player on init is front hand
        for i in [1, 2, 0]:
            self.assertEqual(i, self.round.next_player)
            self.round.step()  # play a card
