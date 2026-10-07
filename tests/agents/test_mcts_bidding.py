import unittest

from skat.agents.bidding import BiddingStyle, ceiling, playable
from skat.agents.mcts import MCTSAgent
from skat.agents.mcts_bidding import MCTSBidder, MCTSBidResult, game_key
from skat.agents.random import RandomAgent
from skat.game import Round
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND


class FakeMCTSBidder:
    def __init__(self, probability: float, suit: int = 2, threshold: float = 0.65):
        self.probability = probability
        self.threshold = threshold
        self.game = SuitGame(suit)

    def evaluate(self, cards, seat, dealer):
        return MCTSBidResult(
            self.game,
            {"diamonds": 0.1, "hearts": 0.2, "spades": 0.7, "clubs": 0.3, "grand": 0.4},
            self.probability,
        )


def round_with(agent, first_hand=None) -> Round:
    first_hand = list(first_hand or FULL_HAND[:10])
    remaining = [card for card in FULL_HAND if card not in first_hand]
    round_ = Round(
        agents=[agent, RandomAgent(seed=1), RandomAgent(seed=2)],
        initial_cards=[first_hand, remaining[:10], remaining[10:20], remaining[20:]],
        start=False,
    )
    round_.deal()
    return round_


class MCTSBidderGateTest(unittest.TestCase):
    def test_rejects_a_game_below_the_confirmation_threshold(self):
        agent = RandomAgent(bidding=BiddingStyle.MCTS, seed=0)
        agent.set_mcts_bidder(FakeMCTSBidder(0.64))
        round_with(agent)

        self.assertEqual(0, agent.max_bid())
        self.assertEqual(0.64, agent.last_bidding_probability)
        self.assertEqual("spades", agent.last_bidding_game)

    def test_accepts_at_threshold_with_a_safe_ceiling(self):
        agent = RandomAgent(bidding=BiddingStyle.MCTS, seed=0)
        bidder = FakeMCTSBidder(0.65)
        round_ = round_with(agent)

        agent.set_mcts_bidder(bidder)
        expected = ceiling(
            BiddingStyle.CONSERVATIVE,
            bidder.game,
            set(round_.player[0].hand),
        )
        self.assertEqual(expected, agent.max_bid())
        self.assertEqual(2, agent.intended_game().suit)

    def test_selected_game_controls_pickup_pressing_and_declaration(self):
        agent = RandomAgent(bidding=BiddingStyle.MCTS, seed=0)
        agent.set_mcts_bidder(FakeMCTSBidder(1.0, suit=1))
        round_ = round_with(agent)
        agent.max_bid()

        self.assertTrue(agent.pickup_skat(round_))
        round_.player[0].receive_cards(round_.skat)
        pressed = agent.press_skat()
        declared = agent.declare_game(round_)

        self.assertEqual(2, len(pressed))
        self.assertEqual(10, len(round_.player[0].hand))
        self.assertEqual(1, declared.suit)

    def test_mcts_style_replaces_the_mcts_agents_playable_gate(self):
        weak = [FULL_HAND[index] for index in (0, 1, 2, 8, 9, 10, 16, 17, 24, 25)]
        self.assertFalse(playable(set(weak)))
        agent = MCTSAgent(bidding=BiddingStyle.MCTS, seed=0)
        agent.set_mcts_bidder(FakeMCTSBidder(1.0))
        round_with(agent, weak)

        self.assertGreater(agent.max_bid(), 0)


class MCTSBidderSearchTest(unittest.TestCase):
    def test_evaluates_five_games_and_rechecks_the_best_on_fresh_worlds(self):
        class ScriptedBidder(MCTSBidder):
            def __init__(self):
                super().__init__(
                    simulations=5,
                    determinizations=1,
                    confirmation_simulations=7,
                    confirmation_determinizations=2,
                    seed=3,
                )
                self.samples = []

            def _sample_worlds(self, cards, soloist, count):
                worlds = [object() for _ in range(count)]
                self.samples.append(worlds)
                return worlds

            def _estimate(self, cards, soloist, dealer, game, worlds, simulations):
                if simulations == self.confirmation_simulations:
                    return 0.71
                return {
                    "diamonds": 0.2,
                    "hearts": 0.3,
                    "spades": 0.8,
                    "clubs": 0.5,
                    "grand": 0.6,
                }[game_key(game)]

        bidder = ScriptedBidder()
        result = bidder.evaluate(set(FULL_HAND[:10]), seat=0, dealer=2)

        self.assertEqual(
            {"diamonds", "hearts", "spades", "clubs", "grand"},
            set(result.candidate_probabilities),
        )
        self.assertEqual("spades", game_key(result.game))
        self.assertEqual(0.71, result.confirmation_probability)
        self.assertIsNot(bidder.samples[0], bidder.samples[1])
        self.assertEqual([1, 2], [len(worlds) for worlds in bidder.samples])

    def test_real_search_is_seed_reproducible(self):
        options = {
            "simulations": 5,
            "determinizations": 1,
            "confirmation_simulations": 5,
            "confirmation_determinizations": 1,
            "seed": 17,
        }
        cards = set(FULL_HAND[:10])
        first = MCTSBidder(**options).evaluate(cards, seat=1, dealer=0)
        second = MCTSBidder(**options).evaluate(cards, seat=1, dealer=0)

        self.assertEqual(game_key(first.game), game_key(second.game))
        self.assertEqual(first.candidate_probabilities, second.candidate_probabilities)
        self.assertEqual(
            first.confirmation_probability, second.confirmation_probability
        )
        self.assertTrue(
            all(0.0 <= value <= 1.0 for value in first.candidate_probabilities.values())
        )


if __name__ == "__main__":
    unittest.main()
