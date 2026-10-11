import unittest

import pytest

from skat.deck import Deck
from skat.game import GamePhase, Round
from skat.games.suit import SuitGame
from skat.models.observation import INPUT_SIZE
from skat.rules import RoundResult


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
        cards = Deck.factory(seed=8).to_list()
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
            self.assertEqual(10, len(player.hand))
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
        torch = pytest.importorskip("torch")
        st = self.round.get_state_t(0)
        self.assertEqual(torch.Size([1, INPUT_SIZE]), st.size())

    def test_reward(self):
        self.assertEqual(0, self.round.cumulative_reward(0))
        self.round.result = RoundResult(0, True, "won")
        self.assertEqual(1, self.round.cumulative_reward(0))
        self.assertEqual(-1, self.round.cumulative_reward(1))
