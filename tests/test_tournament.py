import unittest
from unittest.mock import patch

from skat.agents.random import RandomAgent
from skat.card import Card
from skat.game import Round, SeatScore, Tournament
from skat.hand import FULL_HAND
from skat.rules import RoundResult
from tests.test_bidding import FixedBidAgent


def dealt_hands(seed, rounds=6):
    """
    Seat 0's hand as dealt in each round. Captured at deal time - by the end of
    a round the hands are empty, and trick stacks would vary with the random
    play rather than with the deal.
    """
    hands: list[tuple[str, ...]] = []
    original = Round.deal

    def spy(self) -> None:
        original(self)
        hands.append(tuple(repr(card) for card in self.player[0].hand))

    with patch.object(Round, "deal", spy):
        Tournament(
            rounds=rounds,
            agents=[RandomAgent() for _ in range(3)],
            seed=seed,
        ).start()
    return hands


class SeedingTest(unittest.TestCase):
    def test_each_round_gets_its_own_seed(self) -> None:
        tournament = Tournament(seed=7)
        drawn = [tournament.next_seed() for _ in range(10)]
        self.assertEqual(10, len(set(drawn)))

    def test_the_same_seed_reproduces(self) -> None:
        a = [Tournament(seed=7).next_seed() for _ in range(1)]
        b = [Tournament(seed=7).next_seed() for _ in range(1)]
        self.assertEqual(a, b)

    def test_a_different_seed_differs(self) -> None:
        self.assertNotEqual(
            Tournament(seed=7).next_seed(), Tournament(seed=8).next_seed()
        )

    def test_no_seed_stays_unseeded(self) -> None:
        tournament = Tournament(seed=None)
        self.assertIsNone(tournament.next_seed())

    def test_a_seeded_tournament_varies_its_deals(self) -> None:
        # regression: every round used to be dealt the identical hand, which
        # made a seeded 33-round run about three data points wide
        hands = dealt_hands(seed=11, rounds=6)
        self.assertEqual(6, len(hands))
        self.assertEqual(6, len(set(hands)))

    def test_a_seeded_tournament_is_reproducible(self) -> None:
        self.assertEqual(dealt_hands(seed=11, rounds=4), dealt_hands(seed=11, rounds=4))

    def test_different_seeds_deal_differently(self) -> None:
        self.assertNotEqual(
            dealt_hands(seed=11, rounds=4), dealt_hands(seed=12, rounds=4)
        )

    def test_an_unseeded_tournament_still_deals(self) -> None:
        hands = dealt_hands(seed=None, rounds=4)
        self.assertEqual(4, len(hands))
        self.assertTrue(all(len(hand) == 10 for hand in hands))


class SeatScoreTest(unittest.TestCase):
    def test_totals_add_up(self) -> None:
        score = SeatScore()
        score.solo_rounds, score.solo_wins = 4, 3
        score.defence_rounds, score.defence_wins = 6, 2
        self.assertEqual(10, score.rounds)
        self.assertEqual(5, score.wins)


class FakeRound(Round):
    """
    A finished round reduced to what `Tournament.record` reads. Bypasses
    `Round.__init__` on purpose - seating three agents and dealing 32 cards
    would say nothing extra about the tallying.
    """

    def __init__(self, soloist: int, solo_points: int, overbid: bool = False) -> None:
        self.solo_player_id = soloist
        self.original_deal = None
        self.solo_points = solo_points
        # drive the real `Round.overbid` rather than stubbing it out
        self.game_value = 36
        self.highest_bid = 40 if overbid else 24
        self.result = RoundResult(
            soloist,
            solo_points > 60 and not overbid,
            "overbid" if overbid else "cards",
            score=-80 if overbid else 36,
        )

    @property
    def points_soloist(self) -> int:
        return self.solo_points

    @property
    def points_defenders(self) -> int:
        return 120 - self.solo_points


class RecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tournament = Tournament(agents=[RandomAgent() for _ in range(3)])

    def test_a_soloist_win_credits_only_the_soloist(self) -> None:
        self.tournament.record(FakeRound(1, 80), True)
        seats = self.tournament.seat_scores
        self.assertEqual(
            [1, 1, 80], [seats[1].solo_rounds, seats[1].solo_wins, seats[1].card_points]
        )
        for seat in (0, 2):
            self.assertEqual(1, seats[seat].defence_rounds)
            self.assertEqual(0, seats[seat].defence_wins)
            self.assertEqual(40, seats[seat].card_points)

    def test_a_soloist_loss_credits_both_defenders(self) -> None:
        self.tournament.record(FakeRound(0, 45), False)
        seats = self.tournament.seat_scores
        self.assertEqual(0, seats[0].solo_wins)
        self.assertEqual(1, seats[1].defence_wins)
        self.assertEqual(1, seats[2].defence_wins)
        self.assertEqual([0, 1], self.tournament.scores)

    def test_overbids_are_counted_against_the_soloist(self) -> None:
        self.tournament.record(FakeRound(2, 100, overbid=True), False)
        seats = self.tournament.seat_scores
        self.assertEqual(1, seats[2].overbids)
        self.assertEqual(0, seats[2].solo_wins)
        self.assertEqual([0, 0], [seats[0].overbids, seats[1].overbids])

    def test_a_clean_solo_round_records_no_overbid(self) -> None:
        self.tournament.record(FakeRound(2, 100), True)
        self.assertEqual(0, self.tournament.seat_scores[2].overbids)

    def test_a_passed_round_counts_for_nobody(self) -> None:
        self.tournament.record(FakeRound(-42, 0), None)
        self.assertEqual(1, self.tournament.passed)
        self.assertEqual([0, 0], self.tournament.scores)
        for score in self.tournament.seat_scores:
            self.assertEqual(0, score.rounds)

    def test_role_scores_and_seat_scores_agree(self) -> None:
        self.tournament.record(FakeRound(0, 70), True)
        self.tournament.record(FakeRound(1, 30), False)
        self.tournament.record(FakeRound(2, 90), True)
        seats = self.tournament.seat_scores
        self.assertEqual(self.tournament.scores[0], sum(s.solo_wins for s in seats))
        # each defeat of a soloist is a win for two defenders
        self.assertEqual(
            self.tournament.scores[1] * 2, sum(s.defence_wins for s in seats)
        )


class SummaryTest(unittest.TestCase):
    def test_reports_every_seat_and_its_agent(self) -> None:
        tournament = Tournament(rounds=1, agents=[RandomAgent() for _ in range(3)])
        tournament.record(FakeRound(0, 70), True)
        summary = tournament.summary()
        self.assertEqual(3, summary.count("RandomAgent"))
        self.assertIn("1/1 (100%)", summary)  # the soloist
        self.assertIn("0/1 (0%)", summary)  # the two defenders

    def test_an_unplayed_role_shows_a_dash(self) -> None:
        tournament = Tournament(rounds=0, agents=[RandomAgent() for _ in range(3)])
        self.assertIn("-", tournament.summary())

    def test_counts_passed_rounds(self) -> None:
        tournament = Tournament(rounds=2, agents=[RandomAgent() for _ in range(3)])
        tournament.record(FakeRound(-42, 0), None)
        self.assertIn("1 passed out", tournament.summary())


class OverbidTest(unittest.TestCase):
    """Überreizt (ISkO 5.2.3): a soloist whose game is worth less than the
    bid loses no matter how the cards fell."""

    def round_with(self, highest_bid: int, game_value: int) -> Round:
        round_ = Round(start=False, agents=[RandomAgent() for _ in range(3)])
        round_.highest_bid = highest_bid
        round_.game_value = game_value
        if game_value:
            round_.result = RoundResult(
                0,
                highest_bid <= game_value,
                "overbid" if highest_bid > game_value else "won",
            )
        return round_

    def test_bidding_beyond_the_game_is_an_overbid(self) -> None:
        self.assertTrue(self.round_with(40, 36).overbid)

    def test_bidding_exactly_the_game_value_is_fine(self) -> None:
        self.assertFalse(self.round_with(36, 36).overbid)

    def test_bidding_under_the_game_value_is_fine(self) -> None:
        self.assertFalse(self.round_with(24, 36).overbid)

    def test_undeclared_game_is_never_an_overbid(self) -> None:
        # game_value is 0 until the game is declared
        self.assertFalse(self.round_with(40, 0).overbid)

    def test_skipped_bidding_is_never_an_overbid(self) -> None:
        # ISS replays and most tests never bid, so highest_bid stays 0
        self.assertFalse(self.round_with(0, 36).overbid)


class OverbidRoundTest(unittest.TestCase):
    """
    Same deal twice: seat 1 holds ten of the eleven club trumps and wins the
    cards either way, but loses the round when it bid past what the game is
    worth.
    """

    def deal(self):
        """Seat 1 gets every club plus ♠J and ♥J; ♦J is pinned in seat 0."""
        soloist = [Card(3, rank) for rank in range(8)]  # all clubs
        soloist += [Card(2, 3), Card(1, 3)]  # ♠J, ♥J
        rest = [c for c in FULL_HAND if c not in soloist]
        jack_of_diamonds = Card(0, 3)
        rest.remove(jack_of_diamonds)
        # ♦J to seat 0, never the skat - in the skat it would extend the run of
        # held matadors and change the Spielwert out from under the test
        return [[jack_of_diamonds] + rest[0:9], soloist, rest[9:19], rest[19:21]]

    def play(self, competitor_ceiling: int):
        # seat 1 is front hand (dealer 0) and will hold against seat 2
        max_bids = [0, 120, competitor_ceiling]
        round_ = Round(
            dealer=0,
            skip_bidding=False,
            hand_game=True,
            agents=[FixedBidAgent(m) for m in max_bids],
            start=False,
            initial_cards=self.deal(),
        )
        return round_, round_.start()

    def test_hand_and_earned_schwarz_are_included(self) -> None:
        # mit 3 + Spiel + Hand + earned Schneider + Schwarz -> 7 x 12
        round_, _ = self.play(24)
        self.assertEqual(1, round_.solo_player_id)
        self.assertEqual(84, round_.game_value)

    def test_an_overbid_soloist_loses_despite_winning_the_cards(self) -> None:
        round_, won = self.play(96)  # pushed to 60, game only worth 48
        self.assertEqual(96, round_.highest_bid)
        self.assertTrue(round_.overbid)
        self.assertGreater(round_.points_soloist, round_.points_defenders)
        self.assertFalse(won)

    def test_the_same_hand_wins_when_the_bid_stays_low(self) -> None:
        round_, won = self.play(24)  # 24 is within the game's 48
        self.assertEqual(24, round_.highest_bid)
        self.assertFalse(round_.overbid)
        self.assertGreater(round_.points_soloist, round_.points_defenders)
        self.assertTrue(won)


if __name__ == "__main__":
    unittest.main()
