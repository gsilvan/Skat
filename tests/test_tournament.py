import unittest
from unittest.mock import patch

from skat.agents.random import RandomAgent
from skat.game import Round, SeatScore, Tournament


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

    def __init__(self, soloist: int, solo_points: int) -> None:
        self.solo_player_id = soloist
        self.solo_points = solo_points

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


if __name__ == "__main__":
    unittest.main()
