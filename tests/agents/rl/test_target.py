"""
What the bootstrap target is allowed to look at, and how far it reaches.

Two defects lived here. `optimize_model` took `target_net(s').max(1)[0]` over
all 32 outputs, so the value of a state was the best of every card in the deck
- and a hand holds at most ten, so the winner was nearly always a card the
agent could not play, often one already lying in a finished trick. And
`gamma` was 0.5 over a ten-trick round, which shrank the terminal win bonus to
0.5^9 ~ 0.002 by the first trick: the agent was scored on card points and told
almost nothing about whether it won.
"""

import tempfile
import unittest
from itertools import pairwise

import torch

from skat.agents.random import RandomAgent
from skat.agents.rl import dqn as dqn_module
from skat.game import GamePhase, Round
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND


class FixedNet(torch.nn.Module):
    """A stand-in target net that always answers with the same row.

    A real `nn.Module` rather than a lambda because `DQN.bootstrap` runs it
    through `evaluating`, which sets eval mode - a bare function has none.
    """

    def __init__(self, row: torch.Tensor) -> None:
        super().__init__()
        self.row = row

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.row.expand(states.shape[0], -1).clone()


def bare_dqn(q_row: torch.Tensor):
    """A DQN whose target net answers with `q_row`, and nothing else."""
    dqn = dqn_module.DQN.__new__(dqn_module.DQN)
    dqn.device = torch.device("cpu")
    dqn.target_net = FixedNet(q_row)
    return dqn


def mask_of(*indices: int) -> torch.Tensor:
    mask = torch.zeros(32, dtype=torch.bool)
    for index in indices:
        mask[index] = True
    return mask


class BootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        # card 7 is by far the most valuable, and is the one card not in hand
        q = torch.full((1, 32), -1.0)
        q[0][3] = 2.0
        q[0][5] = 4.0
        q[0][7] = 99.0
        self.dqn = bare_dqn(q)

    def test_an_illegal_card_cannot_set_the_target(self) -> None:
        value = self.dqn.bootstrap(torch.zeros((1, 240)), mask_of(3, 5).unsqueeze(0))
        self.assertEqual(4.0, float(value[0]))

    def test_the_unmasked_max_is_what_it_used_to_be(self) -> None:
        # pins the difference: with everything legal, 99 wins - that was the
        # target for every state regardless of what the agent held
        value = self.dqn.bootstrap(
            torch.zeros((1, 240)), torch.ones(1, 32, dtype=torch.bool)
        )
        self.assertEqual(99.0, float(value[0]))

    def test_a_single_legal_card_gives_its_own_value(self) -> None:
        # the last trick of a round: one card left, no choice to make
        value = self.dqn.bootstrap(torch.zeros((1, 240)), mask_of(3).unsqueeze(0))
        self.assertEqual(2.0, float(value[0]))

    def test_rows_are_masked_independently(self) -> None:
        masks = torch.stack([mask_of(3), mask_of(5), mask_of(7)])
        values = self.dqn.bootstrap(torch.zeros((3, 240)), masks)
        self.assertEqual([2.0, 4.0, 99.0], [float(v) for v in values])

    def test_the_filler_never_becomes_the_value(self) -> None:
        # masked entries are set to finfo.min; if a row's legal cards were ever
        # dropped, the target would be ~-3e38 and the loss would explode
        values = self.dqn.bootstrap(
            torch.zeros((4, 240)), torch.stack([mask_of(3)] * 4)
        )
        self.assertTrue(bool((values > -1e30).all()))

    def test_it_does_not_carry_gradients(self) -> None:
        value = self.dqn.bootstrap(torch.zeros((1, 240)), mask_of(3).unsqueeze(0))
        self.assertFalse(value.requires_grad)

    def test_the_caller_s_mask_is_left_alone(self) -> None:
        # masked_fill_, the in-place form, would corrupt the buffer's own copy
        mask = mask_of(3, 5).unsqueeze(0)
        self.dqn.bootstrap(torch.zeros((1, 240)), mask)
        self.assertEqual(2, int(mask.sum()))


class RewardsTelescopeTest(unittest.TestCase):
    """
    Why gamma is 1.0.

    `DQNAgent` stores the *change* in `Round.cumulative_reward` as each trick's
    reward, so an undiscounted return over a round comes to exactly the final
    outcome - the number the network should be predicting. Any gamma < 1 makes
    the agent prefer card points now over winning later.
    """

    @staticmethod
    def fresh_round() -> Round:
        deal = [FULL_HAND[0:10], FULL_HAND[10:20], FULL_HAND[20:30], FULL_HAND[30:32]]
        round_ = Round(
            skip_bidding=True,
            solo_player_id=0,
            hand_game=True,
            declare_game=SuitGame(3),
            agents=[RandomAgent() for _ in range(3)],
            start=False,
            initial_cards=[list(hand) for hand in deal],
        )
        round_.deal()
        return round_

    @classmethod
    def play(cls) -> tuple[Round, list[float]]:
        round_ = cls.fresh_round()
        round_.phase = GamePhase.PLAYING
        assert round_.game is not None
        rewards, previous = [], round_.cumulative_reward(0)
        while not round_.is_finished:
            round_.step()
            if len(round_.game.trick) == 0:  # a trick just completed
                current = round_.cumulative_reward(0)
                rewards.append(current - previous)
                previous = current
        return round_, rewards

    def test_the_undiscounted_return_is_the_final_outcome(self) -> None:
        round_, rewards = self.play()
        self.assertEqual(10, len(rewards))
        self.assertAlmostEqual(round_.cumulative_reward(0), sum(rewards), places=6)

    def test_discounting_loses_the_outcome(self) -> None:
        # the old gamma, quantified: the terminal win/schneider/schwarz bonus is
        # worth 0.5^9 of itself by the first trick
        round_, rewards = self.play()
        discounted = sum(0.5**step * r for step, r in enumerate(rewards))
        self.assertNotAlmostEqual(round_.cumulative_reward(0), discounted, places=2)
        self.assertLess(0.5**9, 0.01)

    def test_the_reward_starts_from_zero(self) -> None:
        # the telescoping argument needs cumulative_reward(0) == 0 before play
        self.assertEqual(0.0, self.fresh_round().cumulative_reward(0))


class DefaultsTest(unittest.TestCase):
    def test_gamma_is_undiscounted(self) -> None:
        self.assertEqual(1.0, dqn_module.DQN(device="cpu").gamma)


class TransitionTest(unittest.TestCase):
    """End to end: a real agent, a real round, and what lands in the buffer."""

    _tmp: "tempfile.TemporaryDirectory"
    transitions: list

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        from skat.agents.dqn import DQNAgent

        cls._tmp = tempfile.TemporaryDirectory()
        # learn_start is left at its default, so the whole round is warm-up and
        # optimize_model returns before touching the nets
        agent = DQNAgent(
            train=True, device="cpu", path=str(Path(cls._tmp.name) / "unused.pt")
        )
        deal = [FULL_HAND[0:10], FULL_HAND[10:20], FULL_HAND[20:30], FULL_HAND[30:32]]
        round_ = Round(
            skip_bidding=True,
            solo_player_id=0,
            hand_game=True,
            declare_game=SuitGame(3),
            agents=[agent, RandomAgent(), RandomAgent()],
            start=False,
            initial_cards=[list(hand) for hand in deal],
        )
        round_.deal()
        round_.phase = GamePhase.PLAYING
        while not round_.is_finished:
            round_.step()
        cls.transitions = list(agent.dqn.replay_buffer.buffer)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_one_transition_per_trick(self) -> None:
        self.assertEqual(10, len(self.transitions))

    def test_the_mask_shrinks_with_the_hand(self) -> None:
        # nine cards left after the first trick, one after the ninth
        counts = [
            int(t.next_mask.sum()) for t in self.transitions if t.next_mask is not None
        ]
        self.assertEqual(list(range(9, 0, -1)), counts)

    def test_the_masks_are_nested_and_start_from_the_dealt_hand(self) -> None:
        # the point of the fix: a card leaves the mask when it is played and
        # never comes back, so the target can never reach a card already on the
        # table - which, unmasked, is what it almost always reached for
        dealt = set(FULL_HAND[0:10])
        masks = [
            {FULL_HAND[i] for i, on in enumerate(t.next_mask) if on}
            for t in self.transitions
            if t.next_mask is not None
        ]
        self.assertTrue(masks[0] < dealt)
        for earlier, later in pairwise(masks):
            self.assertTrue(later < earlier, f"{later} is not a subset of {earlier}")

    def test_the_terminal_transition_has_neither(self) -> None:
        # they go missing together: nothing to bootstrap from, nothing to mask
        last = self.transitions[-1]
        self.assertIsNone(last.next_state)
        self.assertIsNone(last.next_mask)

    def test_no_earlier_transition_is_terminal(self) -> None:
        for transition in self.transitions[:-1]:
            self.assertIsNotNone(transition.next_state)
            self.assertIsNotNone(transition.next_mask)


if __name__ == "__main__":
    unittest.main()
