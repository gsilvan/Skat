import unittest
from random import Random

from skat.agents import Agent
from skat.agents.bidding import (
    VALID_BIDS,
    BiddingStyle,
    aces,
    best_game,
    ceiling,
    covering_game,
    next_rung,
    playable,
    suit_length,
    trump_count,
)
from skat.card import Card
from skat.games.grand import Grand
from skat.games.suit import SuitGame
from skat.hand import Hand
from skat.player import Player

CLUBS = 3


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def hand_of(*names: str) -> set[Card]:
    return {card(name[0], name[1:]) for name in names}


class StubAgent(Agent):
    """Minimal concrete agent - only the bidding path is under test."""

    def choose_card(self, valid_moves):
        raise NotImplementedError

    def pickup_skat(self, state):
        raise NotImplementedError

    def declare_game(self, state):
        raise NotImplementedError

    def press_skat(self):
        raise NotImplementedError


def seated(cards: set[Card], **kwargs) -> StubAgent:
    agent = StubAgent(**kwargs)
    player = Player(agent, 0)
    player.hand = Hand(cards)
    return agent


# An "ohne 6" club hand: worth 84 at face value, but a top matador arriving in
# the skat cuts the run short, so only 24 is actually guaranteed.
OHNE_SIX = hand_of("♣7", "♣8", "♣9", "♣Q", "♣K")


class CeilingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game = SuitGame(CLUBS)

    def test_conservative_takes_the_guaranteed_value(self) -> None:
        low, _ = self.game.spielwert_range(OHNE_SIX)
        self.assertEqual(low, ceiling(BiddingStyle.CONSERVATIVE, self.game, OHNE_SIX))
        self.assertEqual(24, low)  # ♣J in the skat leaves only mit 1

    def test_nominal_takes_the_hand_at_face_value(self) -> None:
        self.assertEqual(84, ceiling(BiddingStyle.NOMINAL, self.game, OHNE_SIX))

    def test_optimistic_takes_the_best_case(self) -> None:
        _, high = self.game.spielwert_range(OHNE_SIX)
        self.assertEqual(high, ceiling(BiddingStyle.OPTIMISTIC, self.game, OHNE_SIX))
        self.assertGreaterEqual(high, 84)

    def test_the_styles_are_ordered(self) -> None:
        low = ceiling(BiddingStyle.CONSERVATIVE, self.game, OHNE_SIX)
        mid = ceiling(BiddingStyle.NOMINAL, self.game, OHNE_SIX)
        high = ceiling(BiddingStyle.OPTIMISTIC, self.game, OHNE_SIX)
        self.assertLessEqual(low, mid)
        self.assertLessEqual(mid, high)

    def test_random_ignores_the_hand(self) -> None:
        drawn = {
            ceiling(BiddingStyle.RANDOM, self.game, OHNE_SIX, rng=Random(seed))
            for seed in range(40)
        }
        self.assertGreater(len(drawn), 1)
        self.assertTrue(drawn <= set(VALID_BIDS))

    def test_a_mit_hand_can_only_gain_from_the_skat(self) -> None:
        # holding ♣J means the run of held matadors can only grow
        mit = hand_of("♣J", "♣A", "♣X", "♣K")
        low, high = SuitGame(CLUBS).spielwert_range(mit)
        self.assertEqual(low, SuitGame(CLUBS).spielwert(mit))
        self.assertGreater(high, low)

    def test_conservative_never_prices_above_the_worst_case(self) -> None:
        # the property that makes CONSERVATIVE unable to overbid, over a
        # spread of hands
        deck = [Card(s, r) for s in range(4) for r in range(8)]
        rng = Random(0)
        for _ in range(50):
            rng.shuffle(deck)
            cards = set(deck[:10])
            game = best_game(cards)
            bid = ceiling(BiddingStyle.CONSERVATIVE, game, cards)
            worst = min(
                game.spielwert(cards | set(skat))
                for skat in [deck[10:12], deck[12:14], deck[14:16]]
            )
            self.assertLessEqual(bid, worst)


class NextRungTest(unittest.TestCase):
    def test_climbs_the_ladder(self) -> None:
        self.assertEqual(18, next_rung(0, 100))
        self.assertEqual(20, next_rung(18, 100))
        self.assertEqual(22, next_rung(20, 100))

    def test_passes_at_the_ceiling(self) -> None:
        self.assertEqual(0, next_rung(20, 20))

    def test_a_ceiling_on_the_ladder_is_reachable(self) -> None:
        self.assertEqual(63, next_rung(60, 63))

    def test_an_off_ladder_current_bid_does_not_raise(self) -> None:
        # 19 is not a rung; the next one strictly above it is 20
        self.assertEqual(20, next_rung(19, 100))

    def test_the_top_of_the_ladder_terminates(self) -> None:
        self.assertEqual(0, next_rung(264, 10**6))


class AgentBidTest(unittest.TestCase):
    def test_conservative_is_the_default_style(self) -> None:
        self.assertEqual(BiddingStyle.CONSERVATIVE, seated(OHNE_SIX).bidding)

    def test_the_style_drives_the_ceiling(self) -> None:
        self.assertEqual(24, seated(OHNE_SIX).max_bid())
        self.assertEqual(84, seated(OHNE_SIX, bidding=BiddingStyle.NOMINAL).max_bid())

    def test_as_sayer_states_the_next_rung(self) -> None:
        agent = seated(OHNE_SIX, bidding=BiddingStyle.NOMINAL)
        self.assertEqual(18, agent.bid(0))
        self.assertEqual(20, agent.bid(18))

    def test_as_sayer_stops_at_its_ceiling(self) -> None:
        self.assertEqual(0, seated(OHNE_SIX).bid(24))

    def test_as_hearer_holds_within_its_ceiling(self) -> None:
        agent = seated(OHNE_SIX)
        self.assertEqual(22, agent.bid(20, offer=22))
        self.assertEqual(0, agent.bid(24, offer=27))

    def test_the_ceiling_is_cached_within_a_round(self) -> None:
        agent = seated(OHNE_SIX)
        assert agent.state is not None
        self.assertEqual(24, agent.max_bid())
        agent.state.hand = Hand(hand_of("♣J", "♠J", "♥J", "♦J"))
        self.assertEqual(24, agent.max_bid())  # still the cached value
        agent.new_round_event()
        self.assertNotEqual(24, agent.max_bid())  # re-valued

    def test_every_bid_lands_on_the_ladder(self) -> None:
        agent = seated(OHNE_SIX, bidding=BiddingStyle.NOMINAL)
        current = 0
        while (said := agent.bid(current)) != 0:
            self.assertIn(said, VALID_BIDS)
            self.assertGreater(said, current)
            current = said
        self.assertEqual(84, current)


class HandEvaluationTest(unittest.TestCase):
    def test_picks_the_longest_trump_suit(self) -> None:
        game = best_game(hand_of("♦7", "♦8", "♦9", "♦Q", "♦K", "♠A", "♥A"))
        self.assertIsInstance(game, SuitGame)
        self.assertEqual(0, game.suit)

    def test_multiplier_only_breaks_ties(self) -> None:
        game = best_game(hand_of("♠7", "♠8", "♣7", "♣8"))
        self.assertEqual(CLUBS, game.suit)

    def test_jacks_count_towards_every_suit(self) -> None:
        self.assertEqual(3, suit_length(hand_of("♣J", "♠J", "♦7"), 0))

    def test_three_jacks_make_it_a_grand(self) -> None:
        self.assertIsInstance(best_game(hand_of("♣J", "♠J", "♥J", "♦7", "♦8")), Grand)

    def test_a_long_suit_beats_a_jack_heavy_grand(self) -> None:
        cards = hand_of("♣J", "♠J", "♥J", "♦J", "♦A", "♦X", "♦K", "♦Q", "♦9", "♦8")
        self.assertIsInstance(best_game(cards), SuitGame)

    def test_never_chooses_null(self) -> None:
        cards = hand_of("♦7", "♦8", "♦9", "♥7", "♥8", "♠7", "♣7")
        self.assertIsInstance(best_game(cards), SuitGame)

    def test_trump_count_includes_the_jacks(self) -> None:
        cards = hand_of("♣J", "♦J", "♦7", "♦8", "♥A")
        self.assertEqual(4, trump_count(cards, SuitGame(0)))

    def test_aces_are_counted(self) -> None:
        self.assertEqual(2, aces(hand_of("♦A", "♥A", "♠K")))

    def test_five_trumps_are_playable(self) -> None:
        self.assertTrue(playable(hand_of("♦7", "♦8", "♦9", "♦Q", "♦K")))

    def test_four_trumps_need_two_aces(self) -> None:
        self.assertTrue(playable(hand_of("♦7", "♦8", "♦9", "♦Q", "♠A", "♥A")))
        self.assertFalse(playable(hand_of("♦7", "♦8", "♦9", "♦Q", "♠A", "♥7")))

    def test_a_weak_hand_is_not(self) -> None:
        self.assertFalse(playable(hand_of("♦7", "♦8", "♥7", "♠8", "♣9")))


class CoveringGameTest(unittest.TestCase):
    def test_keeps_the_natural_game_when_it_covers(self) -> None:
        cards = hand_of("♦7", "♦8", "♦9", "♦Q", "♦K", "♠A", "♥A")
        self.assertEqual(best_game(cards).suit, covering_game(cards, 18).suit)

    def test_switches_when_the_natural_game_is_too_cheap(self) -> None:
        # a hand whose longest suit is worth less than what was bid
        cards = hand_of("♦7", "♦8", "♦9", "♦Q", "♠J", "♣A", "♣X")
        natural = best_game(cards)
        steep = natural.spielwert(cards) + 1
        chosen = covering_game(cards, steep)
        self.assertGreaterEqual(chosen.spielwert(cards), steep)

    def test_falls_back_to_the_dearest_game_when_nothing_covers(self) -> None:
        cards = hand_of("♦7", "♦8", "♥9", "♠Q", "♣K")
        chosen = covering_game(cards, 10**6)  # unreachable
        best = max(SuitGame(suit).spielwert(cards) for suit in range(4))
        self.assertGreaterEqual(chosen.spielwert(cards), best)

    def test_a_zero_minimum_is_just_the_natural_game(self) -> None:
        # bidding was skipped, so nothing has to be covered
        cards = hand_of("♣7", "♣8", "♣9", "♣Q", "♣K")
        self.assertEqual(best_game(cards).suit, covering_game(cards, 0).suit)


class NeverOverbidsTest(unittest.TestCase):
    """
    The property that gives `CONSERVATIVE` its name, end to end: over many
    deals, a table of conservative bidders is never überreizt. It is not enough
    for the ceiling to be safe - the declared game has to cover the bid too.
    """

    def overbids(self, style: BiddingStyle, rounds: int = 120) -> tuple[int, int]:
        from skat.agents.random import RandomAgent
        from skat.game import Tournament

        tournament = Tournament(
            rounds=rounds,
            agents=[RandomAgent(bidding=style) for _ in range(3)],
            seed=99,
        )
        tournament.start()
        return (
            sum(score.overbids for score in tournament.seat_scores),
            sum(score.solo_rounds for score in tournament.seat_scores),
        )

    def test_conservative_never_overbids(self) -> None:
        overbid, solo = self.overbids(BiddingStyle.CONSERVATIVE)
        self.assertGreater(solo, 0)  # the test would be vacuous otherwise
        self.assertEqual(0, overbid)

    def test_the_riskier_styles_do(self) -> None:
        self.assertGreater(self.overbids(BiddingStyle.OPTIMISTIC)[0], 0)
        self.assertGreater(self.overbids(BiddingStyle.RANDOM)[0], 0)


if __name__ == "__main__":
    unittest.main()
