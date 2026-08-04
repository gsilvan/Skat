import unittest

from skat.agents import Agent
from skat.agents.bidding import BiddingStyle, best_game
from skat.agents.mcts import MCTSAgent
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.game import GamePhase, Round
from skat.games.grand import Grand
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND, Hand
from skat.player import Player

CLUBS = 3


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def hand_of(*names: str) -> set[Card]:
    return {card(name[0], name[1:]) for name in names}


def planned_suit(cards: set[Card]) -> int:
    """The trump suit `best_game` settled on, asserting it chose a suit game."""
    game = best_game(cards)
    assert isinstance(game, SuitGame)
    return game.suit


def seated(agent: MCTSAgent, cards) -> Player:
    """Give the agent a `Player` to look at, without a round behind it."""
    player = Player(agent, 0)
    player.hand = Hand(cards)
    return player


def make_round(agent: MCTSAgent, seat: int = 0, soloist: int = 0) -> Round:
    """A round dealt from a fixed 32-card split, paused at the first trick."""
    agents: list[Agent] = [RandomAgent(), RandomAgent(), RandomAgent()]
    agents[seat] = agent
    deal = [FULL_HAND[0:10], FULL_HAND[10:20], FULL_HAND[20:30], FULL_HAND[30:32]]
    round_ = Round(
        skip_bidding=True,
        solo_player_id=soloist,
        hand_game=True,
        declare_game=SuitGame(CLUBS),
        agents=agents,
        start=False,
        initial_cards=deal,
    )
    round_.deal()
    round_.phase = GamePhase.PLAYING
    return round_


class SkatTest(unittest.TestCase):
    def test_takes_the_skat_on_an_ordinary_hand(self) -> None:
        agent = MCTSAgent(seed=0)
        seated(agent, hand_of("♦7", "♦8", "♦9", "♦Q", "♦K", "♥A"))
        self.assertTrue(agent.pickup_skat(None))

    def test_plays_a_hand_game_on_a_monster(self) -> None:
        agent = MCTSAgent(seed=0)
        seated(
            agent,
            hand_of("♣J", "♠J", "♥J", "♦J", "♦A", "♦X", "♦K", "♦Q", "♦9", "♦8"),
        )
        self.assertFalse(agent.pickup_skat(None))

    def test_presses_two_non_trump_cards(self) -> None:
        agent = MCTSAgent(seed=0)
        player = seated(
            agent,
            hand_of(
                "♣7",
                "♣8",
                "♣9",
                "♣Q",
                "♣K",
                "♣X",  # trumps
                "♦X",  # singleton, fat
                "♥7",
                "♥8",
                "♠7",
                "♠8",
                "♠9",
            ),
        )
        pressed = agent.press_skat()
        trumps = set(SuitGame.trump_cards(CLUBS))
        self.assertEqual(2, len(pressed))
        self.assertEqual(10, len(player.hand))
        self.assertEqual(set(), set(pressed) & trumps)
        # the fat singleton is both the void and the points worth hiding
        self.assertIn(card("♦", "X"), pressed)

    def test_declares_what_it_planned(self) -> None:
        agent = MCTSAgent(seed=0)
        cards = hand_of("♦7", "♦8", "♦9", "♦Q", "♦K", "♥A")
        seated(agent, cards)
        declared = agent.declare_game(None)
        assert isinstance(declared, SuitGame)
        self.assertEqual(planned_suit(cards), declared.suit)


class DeterminizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = MCTSAgent(simulations=20, determinizations=2, seed=0)
        self.round = make_round(self.agent)

    def test_world_matches_the_visible_hand_sizes(self) -> None:
        state = self.agent.determinize()
        self.assertEqual(list(self.round.player[0].hand), state.hands[0])
        for seat in (1, 2):
            self.assertEqual(len(self.round.player[seat].hand), len(state.hands[seat]))

    def test_world_deals_every_card_at_most_once(self) -> None:
        state = self.agent.determinize()
        dealt = [c for hand in state.hands for c in hand]
        self.assertEqual(len(dealt), len(set(dealt)))

    def test_soloist_is_credited_with_the_skat_points(self) -> None:
        # hand game: the skat is unknown, but its points still count
        state = self.agent.determinize()
        self.assertEqual(
            120,
            sum(c.value for hand in state.hands for c in hand)
            + state.points[state.soloist],
        )

    def test_worlds_differ(self) -> None:
        worlds = {tuple(self.agent.determinize().hands[1]) for _ in range(10)}
        self.assertGreater(len(worlds), 1)

    def test_played_cards_never_reappear(self) -> None:
        for _ in range(7):  # two full tricks plus part of a third
            self.round.step()
        played = {c for t in self.round.trick_history.buffer for _, c in t.buffer}
        assert self.round.game is not None
        played |= {c for _, c in self.round.game.trick.buffer}
        state = self.agent.determinize()
        for hand in state.hands:
            self.assertEqual(set(), played & set(hand))

    def test_world_is_positioned_at_our_turn(self) -> None:
        state = self.agent.determinize()
        self.assertEqual(0, state.to_move)
        self.assertEqual(self.round.solo_player_id, state.soloist)


class ChooseCardTest(unittest.TestCase):
    def test_plays_a_legal_card_and_gives_it_up(self) -> None:
        agent = MCTSAgent(simulations=20, determinizations=2, seed=0)
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        choice = agent.choose_card(valid)
        self.assertIn(choice, valid)
        self.assertNotIn(choice, round_.player[0].hand)
        self.assertEqual(9, len(round_.player[0].hand))

    def test_a_single_option_skips_the_search(self) -> None:
        agent = MCTSAgent(simulations=10**9, determinizations=10**9, seed=0)
        round_ = make_round(agent)
        only = round_.player[0].hand[0]
        self.assertEqual(only, agent.search({only}))  # would hang if searched

    def test_plays_a_full_round_against_random_agents(self) -> None:
        agent = MCTSAgent(simulations=20, determinizations=2, seed=0)
        round_ = make_round(agent)
        round_.start()
        self.assertTrue(round_.is_finished)
        self.assertEqual(120, round_.points_soloist + round_.points_defenders)

    def test_three_mcts_agents_finish_a_round(self) -> None:
        agents = [
            MCTSAgent(simulations=20, determinizations=2, seed=i) for i in range(3)
        ]
        round_ = Round(skip_bidding=False, agents=agents, start=False)
        result = round_.start()
        self.assertIn(result, (True, False, None))
        if result is not None:
            self.assertEqual(120, round_.points_soloist + round_.points_defenders)


if __name__ == "__main__":
    unittest.main()
