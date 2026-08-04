"""
Agents that are not human must never reach for stdin.

`DQNAgent` used to subclass `CommandLineAgent` to borrow prompts for the
decisions the network does not make, so an unattended training run blocked on
`input()` at the very first bid - before a single card was played. These tests
pin the fix from both sides: the interactive methods are gone from the class,
and a full round runs with stdin closed.
"""

import io
import unittest
from unittest.mock import patch

from skat.agents import Agent
from skat.agents.bidding import BiddingStyle
from skat.agents.command_line import CommandLineAgent
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.dqn import DQNAgent
from skat.agents.mcts import MCTSAgent
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.game import Round
from skat.games.grand import Grand
from skat.games.suit import SuitGame
from skat.hand import Hand
from skat.player import Player

AUTOMATIC_AGENTS = (DQNAgent, MCTSAgent, RandomAgent)
#: Every decision a soloist has to make outside of playing cards.
NON_CARD_DECISIONS = ("bid", "pickup_skat", "press_skat", "declare_game")


def card(suit: str, rank: str) -> Card:
    return Card("♦♥♠♣".index(suit), "789JQKXA".index(rank))


def hand_of(*names: str) -> set[Card]:
    return {card(name[0], name[1:]) for name in names}


class NoAgentPromptsForTrainingTest(unittest.TestCase):
    def test_the_interactive_agent_is_not_in_the_ancestry(self) -> None:
        for agent in AUTOMATIC_AGENTS:
            self.assertNotIn(CommandLineAgent, agent.__mro__, agent.__name__)

    def test_no_decision_resolves_to_a_prompt(self) -> None:
        for agent in AUTOMATIC_AGENTS:
            for decision in NON_CARD_DECISIONS:
                self.assertIsNot(
                    getattr(agent, decision),
                    getattr(CommandLineAgent, decision, None),
                    f"{agent.__name__}.{decision} still prompts",
                )

    def test_bidding_comes_from_the_shared_mechanics(self) -> None:
        # not overridden anywhere, so BiddingStyle is what decides
        self.assertIs(DQNAgent.bid, Agent.bid)

    def test_skat_handling_comes_from_the_mixin(self) -> None:
        for decision in ("pickup_skat", "press_skat", "declare_game"):
            self.assertIs(
                getattr(DQNAgent, decision), getattr(HeuristicDeclarer, decision)
            )

    def test_the_command_line_agent_still_prompts(self) -> None:
        # the fix must not have flattened the human agent by accident
        for decision in NON_CARD_DECISIONS:
            self.assertIn(
                "input", getattr(CommandLineAgent, decision).__code__.co_names
            )


class HeuristicDeclarerTest(unittest.TestCase):
    """The mixin itself, exercised without any network or search."""

    class Declarer(HeuristicDeclarer, Agent):
        def choose_card(self, valid_moves):
            choice = next(iter(valid_moves))
            self.state.hand.remove(choice)
            return choice

    def seated(self, cards) -> "HeuristicDeclarerTest.Declarer":
        agent = self.Declarer()
        player = Player(agent, 0)
        player.hand = Hand(cards)
        return agent

    def test_conservative_is_the_default_style(self) -> None:
        self.assertEqual(BiddingStyle.CONSERVATIVE, self.seated(hand_of("♦7")).bidding)

    def test_it_bids_without_being_asked(self) -> None:
        agent = self.seated(hand_of("♦7", "♦8", "♦9", "♦Q", "♦K"))
        self.assertGreater(agent.max_bid(), 0)
        self.assertEqual(18, agent.bid(0))

    def test_it_takes_the_skat_on_an_ordinary_hand(self) -> None:
        self.assertTrue(
            self.seated(hand_of("♦7", "♦8", "♦9", "♦Q", "♦K")).pickup_skat(None)
        )

    def test_it_plays_a_hand_game_on_a_monster(self) -> None:
        monster = hand_of("♣J", "♠J", "♥J", "♦J", "♦A", "♦X", "♦K", "♦Q", "♦9", "♦8")
        self.assertFalse(self.seated(monster).pickup_skat(None))

    def test_pressing_drops_two_non_trumps(self) -> None:
        agent = self.seated(
            hand_of(
                "♣7", "♣8", "♣9", "♣Q", "♣K", "♣X", "♦X", "♥7", "♥8", "♠7", "♠8", "♠9"
            )
        )
        pressed = agent.press_skat()
        assert agent.state is not None
        self.assertEqual(2, len(pressed))
        self.assertEqual(10, len(agent.state.hand))
        self.assertEqual(set(), set(pressed) & set(SuitGame.trump_cards(3)))

    def test_declaring_agrees_with_the_intended_game(self) -> None:
        cards = hand_of("♣J", "♠J", "♥J", "♦7", "♦8")
        agent = self.seated(cards)
        self.assertIsInstance(agent.intended_game(), Grand)
        self.assertIsInstance(agent.declare_game(None), Grand)


class RunsWithoutStdinTest(unittest.TestCase):
    def test_a_full_round_needs_no_input(self) -> None:
        """
        Any read from stdin raises instead of blocking, so a hung prompt shows
        up as a failure here rather than as a training run that never starts.
        """

        class Exploding(io.IOBase):
            def readline(self, *a, **k):
                raise AssertionError("agent tried to read stdin")

            def read(self, *a, **k):
                raise AssertionError("agent tried to read stdin")

        agents = [HeuristicDeclarerTest.Declarer() for _ in range(3)]
        with patch("sys.stdin", Exploding()):
            round_ = Round(skip_bidding=False, agents=agents, start=False)
            result = round_.start()

        self.assertIn(result, (True, False, None))
        if result is not None:
            self.assertEqual(120, round_.points_soloist + round_.points_defenders)


if __name__ == "__main__":
    unittest.main()
