import unittest
from random import Random

from skat.agents.search.playout import PlayoutState
from skat.agents.search.uct import uct_search
from skat.card import Card
from skat.games.suit import SuitGameTrick

CLUBS = 3


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


class UctSearchTest(unittest.TestCase):
    def endgame(self) -> PlayoutState:
        """
        Seat 2 has led ♦A and seat 0 (the soloist, on 45 points) is void in
        diamonds. Trumping with ♣7 collects the 21 points in the trick and
        carries the soloist over 61; discarding ♥7 hands them to the defence.
        Both continuations are forced, so the value gap is unambiguous.
        """
        trick = SuitGameTrick(CLUBS)
        trick.append(2, card("♦", "A"))
        return PlayoutState(
            trick=trick,
            hands=[
                [card("♣", "7"), card("♥", "7")],
                [card("♦", "X"), card("♥", "8")],
                [card("♥", "9")],
            ],
            to_move=0,
            points=[45, 54, 0],
            soloist=0,
        )

    def test_finds_the_winning_trump(self) -> None:
        stats = uct_search(self.endgame(), iterations=60, rng=Random(0))
        best = max(stats, key=lambda move: stats[move][1] / stats[move][0])
        self.assertEqual(card("♣", "7"), best)

    def test_values_separate_win_from_loss(self) -> None:
        stats = uct_search(self.endgame(), iterations=60, rng=Random(0))
        trumped = stats[card("♣", "7")]
        discarded = stats[card("♥", "7")]
        self.assertGreater(trumped[1] / trumped[0], 0.9)
        self.assertLess(discarded[1] / discarded[0], 0.1)

    def test_can_report_strict_win_probabilities_for_bidding(self) -> None:
        stats = uct_search(
            self.endgame(), iterations=60, rng=Random(0), outcome_only=True
        )
        trumped = stats[card("♣", "7")]
        discarded = stats[card("♥", "7")]
        self.assertEqual(1.0, trumped[1] / trumped[0])
        self.assertEqual(0.0, discarded[1] / discarded[0])

    def test_spends_more_visits_on_the_better_move(self) -> None:
        stats = uct_search(self.endgame(), iterations=200, rng=Random(0))
        self.assertGreater(stats[card("♣", "7")][0], stats[card("♥", "7")][0])

    def test_reports_every_root_move(self) -> None:
        state = self.endgame()
        stats = uct_search(state, iterations=40, rng=Random(1))
        self.assertEqual(set(state.legal_moves()), set(stats))

    def test_does_not_mutate_the_root_state(self) -> None:
        state = self.endgame()
        uct_search(state, iterations=40, rng=Random(2))
        self.assertEqual(2, len(state.hands[0]))
        self.assertEqual([45, 54, 0], state.points)
        self.assertEqual(1, len(state.trick))

    def test_visits_are_accounted_for(self) -> None:
        stats = uct_search(self.endgame(), iterations=50, rng=Random(3))
        self.assertEqual(50, sum(visits for visits, _ in stats.values()))

    def test_defender_maximises_against_the_soloist(self) -> None:
        # Mirror image: the soloist (seat 2, on 45) led ♦A and defender seat 0
        # is void. Trumping keeps the 21 points away from the soloist, so the
        # defender must trump for exactly the opposite reason.
        trick = SuitGameTrick(CLUBS)
        trick.append(2, card("♦", "A"))
        state = PlayoutState(
            trick=trick,
            hands=[
                [card("♣", "7"), card("♥", "7")],
                [card("♦", "X"), card("♥", "8")],
                [card("♥", "9")],
            ],
            to_move=0,
            points=[54, 0, 45],
            soloist=2,
        )
        stats = uct_search(state, iterations=100, rng=Random(0))
        best = max(stats, key=lambda move: stats[move][1] / stats[move][0])
        self.assertEqual(card("♣", "7"), best)


if __name__ == "__main__":
    unittest.main()
