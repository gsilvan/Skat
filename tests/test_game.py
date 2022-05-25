import unittest
from unittest.mock import PropertyMock, patch

import torch

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
        # game with unclear positions
        custom_round = Round(start=False)
        self.assertEqual(0, custom_round.points_soloist)
        # TODO: check points after some cards were played

    def test_points_defenders(self) -> None:
        self.assertEqual(0, self.round.points_defenders)
        # game with unclear positions
        custom_round = Round(start=False)
        self.assertEqual(0, custom_round.points_defenders)
        # TODO: check points after some cards were played

    def test_deal_from_deck(self) -> None:
        self.assertEqual(0, len(self.round.deck))
        self.round.deal()
        for player in self.round.player:
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
        for player in custom_round.player:
            self.assertEqual(0, len(player.hand))
        self.assertEqual(0, len(custom_round.skat))
        custom_round.deal()
        for player in custom_round.player:
            self.assertEqual(3, len(player.hand))
        self.assertEqual(2, len(custom_round.skat))

    def test_next_player(self) -> None:
        self.round.deal()
        self.round.phase = GamePhase.PLAYING
        # next_player on init is front hand
        for i in [1, 2, 0]:
            self.assertEqual(i, self.round.next_player)
            self.round.step()  # play a card
        # no declared game
        custom_round = Round(start=False, dealer=1)
        self.assertEqual(2, custom_round.next_player)

    def test_get_state_t(self) -> None:
        self.round.deal()
        st: torch.Tensor = self.round.get_state_t(0)
        self.assertEqual(torch.Size([1, 139]), st.size())

    def test_reward(self) -> None:
        soloist_id = 0
        self.round.solo_player_id = soloist_id

        with patch(
            "skat.game.Round.points_soloist",
            new_callable=PropertyMock,
            return_value=102,
        ):
            self.assertAlmostEqual(
                0.8699, self.round.cumulative_reward(soloist_id), places=3
            )

        with patch(
            "skat.game.Round.points_defenders",
            new_callable=PropertyMock,
            return_value=20,
        ):
            self.assertAlmostEqual(
                0.0333332, self.round.cumulative_reward((soloist_id + 1) % 3), places=3
            )
