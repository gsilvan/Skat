"""
What the student is allowed to play, and how loudly it fails.

A `PolicyAgent` is 32 logits and an argmax, so the only thing standing between
it and an illegal card is the mask - there is no search to reject the move and
no legality check further down. That is the first half of this file.

The second half is about silence. The agent never trains, so every way it can
be wrong is a way it can be wrong quietly: a missing checkpoint that becomes a
randomly initialised network, or a DQN checkpoint of the identical shape whose
32 outputs are Q-values rather than policy logits. Either would play a full
tournament and produce a plausible-looking table.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from skat.agents import Agent
from skat.agents.distill.checkpoint import save_policy
from skat.agents.policy import PolicyAgent
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.game import GamePhase, Round
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND
from skat.models.card_transformer import CardTransformer
from skat.models.policy import PolicyNet
from skat.models.q_net import QNet

CLUBS = 3


def make_round(agent: Agent, seat: int = 0, soloist: int = 0) -> Round:
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
        initial_cards=[list(hand) for hand in deal],
    )
    round_.deal()
    round_.phase = GamePhase.PLAYING
    return round_


def favouring(*slots: int):
    """A stand-in network that ranks `slots` above everything else."""

    def forward(state: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((state.shape[0], 32))
        for rank, slot in enumerate(slots):
            logits[:, slot] = 100.0 - rank
        return logits

    return forward


class PolicyAgentTest(unittest.TestCase):
    """Every case here needs a checkpoint on disk, since the agent refuses to
    exist without one."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.directory.name, "policy.pt")
        torch.manual_seed(0)
        save_policy(cls.path, PolicyNet())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def agent(self, **kwargs) -> PolicyAgent:
        kwargs.setdefault("path", self.path)
        kwargs.setdefault("device", "cpu")
        return PolicyAgent(**kwargs)


class LegalityTest(PolicyAgentTest):
    def setUp(self) -> None:
        self.agent_ = self.agent()
        self.round = make_round(self.agent_)
        assert self.round.game is not None
        self.valid = self.round.player[0].valid_moves(self.round.game.trick)

    def test_it_never_plays_a_card_it_does_not_hold(self) -> None:
        # The network's favourite is a card in someone else's hand; the mask is
        # the only thing that stops it being played.
        outsider = FULL_HAND[25]
        self.assertNotIn(outsider, self.round.player[0].hand)
        self.agent_.net = favouring(outsider.np_index)
        self.assertIn(self.agent_.select(self.valid), self.valid)

    def test_it_plays_the_best_legal_card_when_its_favourite_is_illegal(self) -> None:
        wanted = sorted(self.valid, key=lambda c: c.np_index)[1]
        self.agent_.net = favouring(FULL_HAND[25].np_index, wanted.np_index)
        self.assertEqual(wanted, self.agent_.select(self.valid))

    def test_it_gives_the_card_up(self) -> None:
        choice = self.agent_.choose_card(self.valid)
        self.assertIn(choice, self.valid)
        self.assertNotIn(choice, self.round.player[0].hand)
        self.assertEqual(9, len(self.round.player[0].hand))

    def test_a_single_option_skips_the_forward(self) -> None:
        # Matching the corpus, which holds no forced decisions at all.
        def explode(_):
            raise AssertionError("asked the network about a forced move")

        self.agent_.net = explode
        only = self.round.player[0].hand[0]
        self.assertEqual(only, self.agent_.select({only}))

    def test_it_plays_a_full_round_against_random_agents(self) -> None:
        round_ = make_round(self.agent())
        round_.start()
        self.assertTrue(round_.is_finished)
        self.assertEqual(120, round_.points_soloist + round_.points_defenders)


class DeterminismTest(PolicyAgentTest):
    def test_it_is_deterministic_at_temperature_zero(self) -> None:
        agent = self.agent()
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        self.assertEqual(1, len({agent.select(valid) for _ in range(5)}))

    def test_two_agents_from_one_checkpoint_agree(self) -> None:
        # The premise of a single `--policy` file serving all three seats.
        first, second = self.agent(), self.agent()
        round_ = make_round(first)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        second.set_state(round_.player[0])
        self.assertEqual(first.select(valid), second.select(valid))

    def test_temperature_zero_touches_no_rng(self) -> None:
        agent = self.agent(seed=1)
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        with patch("torch.multinomial", side_effect=AssertionError("sampled")):
            agent.select(valid)


class SamplingTest(PolicyAgentTest):
    def test_sampling_only_ever_draws_a_legal_card(self) -> None:
        agent = self.agent(temperature=1.0, seed=0)
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        for _ in range(50):
            self.assertIn(agent.select(valid), valid)

    def test_it_does_not_always_pick_the_argmax(self) -> None:
        # Otherwise `--policy-temperature` is a no-op and the test above is
        # only re-testing the greedy path.
        agent = self.agent(temperature=2.0, seed=0)
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        self.assertGreater(len({agent.select(valid) for _ in range(30)}), 1)

    def test_a_seed_makes_the_sampling_reproducible(self) -> None:
        def draws(seed: int) -> list[Card]:
            agent = self.agent(temperature=1.0, seed=seed)
            round_ = make_round(agent)
            assert round_.game is not None
            valid = round_.player[0].valid_moves(round_.game.trick)
            return [agent.select(valid) for _ in range(10)]

        self.assertEqual(draws(3), draws(3))


class RefusalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_a_missing_checkpoint_is_refused_with_the_training_command(self) -> None:
        missing = os.path.join(self.directory.name, "nope.pt")
        with self.assertRaises(FileNotFoundError) as caught:
            PolicyAgent(path=missing, device="cpu")
        message = str(caught.exception)
        self.assertIn(missing, message)
        self.assertIn("skat.agents.distill.train", message)
        self.assertIn("--record-mcts", message)

    def test_a_dqn_checkpoint_is_refused(self) -> None:
        # Same 240 -> 32 shape, different meaning. Loading it would give an
        # agent that plays Q-values as policy logits, silently.
        path = os.path.join(self.directory.name, "checkpoint_p0.pt")
        torch.save(QNet().state_dict(), path)
        with self.assertRaises(RuntimeError) as caught:
            PolicyAgent(path=path, device="cpu")
        self.assertIn("Q-values", str(caught.exception))


class TransformerSeatTest(PolicyAgentTest):
    """
    `PolicyAgent` was not touched to add a second architecture - dispatch lives
    in the checkpoint, and the agent's only coupling is `net(state) -> logits`.
    This class is what proves that: the same cases, a different network behind
    the same file.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.directory.name, "transformer.pt")
        torch.manual_seed(0)
        save_policy(cls.path, CardTransformer())

    def test_it_loads_as_a_transformer(self) -> None:
        self.assertIsInstance(self.agent().net, CardTransformer)

    def test_it_plays_a_full_round_against_random_agents(self) -> None:
        round_ = make_round(self.agent())
        round_.start()
        self.assertTrue(round_.is_finished)
        self.assertEqual(120, round_.points_soloist + round_.points_defenders)

    def test_it_never_plays_a_card_it_does_not_hold(self) -> None:
        agent = self.agent()
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        outsider = FULL_HAND[25]
        self.assertNotIn(outsider, round_.player[0].hand)
        agent.net = favouring(outsider.np_index)
        self.assertIn(agent.select(valid), valid)

    def test_it_is_deterministic_at_temperature_zero(self) -> None:
        agent = self.agent()
        round_ = make_round(agent)
        assert round_.game is not None
        valid = round_.player[0].valid_moves(round_.game.trick)
        self.assertEqual(1, len({agent.select(valid) for _ in range(5)}))


class DeclaringTest(PolicyAgentTest):
    """Card play is all that was distilled; the rest is inherited."""

    def test_it_bids_and_declares_without_a_network(self) -> None:
        agent = self.agent()
        round_ = Round(skip_bidding=False, agents=[agent, RandomAgent(), RandomAgent()])
        self.assertIn(round_.solo_player_id, (-42, 0, 1, 2))

    def test_the_skat_phase_comes_from_the_heuristic_declarer(self) -> None:
        from skat.agents.declaring import HeuristicDeclarer

        self.assertIsInstance(self.agent(), HeuristicDeclarer)


if __name__ == "__main__":
    unittest.main()
