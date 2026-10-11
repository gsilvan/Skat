import unittest

from skat.agents import Agent
from skat.game import GamePhase, Round
from skat.games.suit import SuitGame


class FixedBidAgent(Agent):
    """Test double: bids up to a fixed ceiling instead of valuing a hand, so
    the shared Sagen/Hören mechanics in `Agent.bid` (ISkO §3.3.2) are what gets
    exercised here."""

    def __init__(self, max_bid: int) -> None:
        super().__init__()
        self.ceiling = max_bid

    def max_bid(self) -> int:
        return self.ceiling

    def pickup_skat(self, state=None) -> bool:
        return True

    def press_skat(self):
        return [self.state.hand[0], self.state.hand[1]]

    def declare_game(self, state=None):
        return SuitGame(3)

    def choose_card(self, valid_moves):
        card = min(valid_moves, key=lambda c: c.np_index)
        return card


def make_round(max_bids: tuple[int, int, int], dealer: int = 0) -> Round:
    """`max_bids` is indexed by seat_id (0, 1, 2), not by hand position."""
    r = Round(
        dealer=dealer,
        skip_bidding=False,
        start=False,
        agents=[FixedBidAgent(m) for m in max_bids],
    )
    r.deal()
    return r


class BiddingTest(unittest.TestCase):
    def setUp(self) -> None:
        # dealer=0 -> front_hand=1, middle_hand=2, back_hand=0
        self.front, self.middle, self.back = 1, 2, 0

    def test_middle_wins_against_front(self) -> None:
        max_bids = [0, 0, 0]
        max_bids[self.middle] = 100
        max_bids[self.front] = 20
        max_bids[self.back] = 0
        r = make_round(max_bids)
        r.bidding()
        self.assertEqual(self.middle, r.solo_player_id)
        self.assertEqual(22, r.highest_bid)  # front's ceiling (20) is exceeded

    def test_front_holds_and_wins_against_middle(self) -> None:
        max_bids = [0, 0, 0]
        max_bids[self.middle] = 20
        max_bids[self.front] = 100
        max_bids[self.back] = 0
        r = make_round(max_bids)
        r.bidding()
        self.assertEqual(self.front, r.solo_player_id)
        self.assertEqual(20, r.highest_bid)  # middle's ceiling

    def test_back_hand_beats_stage_one_winner(self) -> None:
        max_bids = [0, 0, 0]
        max_bids[self.middle] = 22  # wins stage one at 20 (front's ceiling)
        max_bids[self.front] = 18
        max_bids[self.back] = 24  # exceeds middle's ceiling (22)
        r = make_round(max_bids)
        r.bidding()
        self.assertEqual(self.back, r.solo_player_id)
        self.assertEqual(23, r.highest_bid)  # Null 23 is the next valid rung after 22

    def test_stage_one_winner_retains_against_weak_back_hand(self) -> None:
        max_bids = [0, 0, 0]
        max_bids[self.middle] = 22  # wins stage one at 20 (front's ceiling)
        max_bids[self.front] = 18
        max_bids[self.back] = 20  # can't exceed the held value of 20
        r = make_round(max_bids)
        r.bidding()
        self.assertEqual(self.middle, r.solo_player_id)
        self.assertEqual(20, r.highest_bid)

    def test_front_hand_claims_unprompted_when_nobody_else_bids(self) -> None:
        # ISkO §3.3.6: if middle and back offer nothing, front may still play.
        max_bids = [0, 0, 0]
        max_bids[self.front] = 18
        r = make_round(max_bids)
        r.bidding()
        self.assertEqual(self.front, r.solo_player_id)
        self.assertEqual(18, r.highest_bid)

    def test_eingepasst_when_nobody_bids(self) -> None:
        # ISkO §3.2.19, §3.3.7: a valid outcome, not an error.
        r = make_round((0, 0, 0))
        r.bidding()
        self.assertEqual(-42, r.solo_player_id)
        self.assertEqual(0, r.highest_bid)

    def test_eingepasst_does_not_crash_start(self) -> None:
        # regression test: this used to raise IndexError (self.player[-42]).
        r = Round(
            dealer=0,
            skip_bidding=False,
            start=False,
            agents=[FixedBidAgent(0), FixedBidAgent(0), FixedBidAgent(0)],
        )
        result = r.start()
        self.assertIsNone(result)
        self.assertEqual(-42, r.solo_player_id)
        self.assertEqual(GamePhase.EINGEPASST, r.phase)

    def test_back_hand_indexerror_regression(self) -> None:
        # regression test: middle passes immediately, back then bids - used to
        # index self.player[-42] and raise IndexError.
        max_bids = [0, 0, 0]
        max_bids[self.back] = 40
        r = make_round(max_bids)
        r.bidding()  # must not raise
        self.assertEqual(self.back, r.solo_player_id)
        self.assertEqual(18, r.highest_bid)


if __name__ == "__main__":
    unittest.main()


def test_official_ladder_includes_null_and_excludes_impossible_values():
    assert {18, 23, 35, 46, 59, 264} <= set(Agent.VALID_BIDS)
    assert not {19, 250, 253, 260, 261} & set(Agent.VALID_BIDS)
