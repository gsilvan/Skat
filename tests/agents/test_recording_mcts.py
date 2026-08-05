"""
What the teacher writes down, and what it deliberately doesn't.

`RecordingMCTSAgent` is the whole data-collection half of the distillation: an
hour of MCTS goes in and a corpus comes out, and if the corpus is subtly wrong
nothing downstream will say so - the student will just train to a worse ceiling
for reasons nobody can point at. So the invariants are pinned here rather than
inferred from a training curve.

The sharpest one is that no visit count may land on a card that was not a legal
move. `MCTSAgent.tally` uses `totals.setdefault`, so a determinization that
somehow offered a move the real position does not have would end up in the row
and the student would learn to play cards it does not hold.
"""

import unittest
from unittest.mock import patch

import numpy as np

from skat.agents import Agent
from skat.agents.mcts import MCTSAgent, RecordingMCTSAgent
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.game import GamePhase, Round
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND

CLUBS = 3


class Recorder:
    """A `ShardWriter` reduced to what `RecordingMCTSAgent` actually calls."""

    def __init__(self) -> None:
        self.rounds = 0
        self.samples: list[dict] = []

    def new_round(self) -> None:
        self.rounds += 1

    def add(self, **sample) -> None:
        sample["group"] = self.rounds
        self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)


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


def recording_agent(**kwargs) -> tuple[RecordingMCTSAgent, Recorder]:
    kwargs.setdefault("simulations", 20)
    kwargs.setdefault("determinizations", 2)
    kwargs.setdefault("seed", 0)
    recorder = Recorder()
    return RecordingMCTSAgent(recorder=recorder, **kwargs), recorder


class CountingRecordingAgent(RecordingMCTSAgent):
    """Counts the decisions it was asked for, and how many were forced."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decisions = 0
        self.forced = 0

    def search(self, valid_moves):
        self.decisions += 1
        self.forced += len(valid_moves) == 1
        return super().search(valid_moves)


def counting_agent(**kwargs) -> tuple[CountingRecordingAgent, Recorder]:
    kwargs.setdefault("simulations", 20)
    kwargs.setdefault("determinizations", 2)
    kwargs.setdefault("seed", 0)
    recorder = Recorder()
    return CountingRecordingAgent(recorder=recorder, **kwargs), recorder


class RecordedRowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent, self.recorder = recording_agent()
        self.round = make_round(self.agent)
        assert self.round.game is not None
        self.valid = self.round.player[0].valid_moves(self.round.game.trick)

    def test_the_recorded_moves_are_exactly_the_legal_ones(self) -> None:
        self.agent.search(self.valid)
        self.assertEqual(self.valid, set(self.recorder.samples[0]["moves"]))

    def test_no_visit_count_ever_lands_on_an_illegal_card(self) -> None:
        self.agent.search(self.valid)
        totals = self.recorder.samples[0]["totals"]
        self.assertEqual(self.valid, set(totals))
        for move, (visits, _) in totals.items():
            self.assertIn(move, self.valid)
            self.assertGreaterEqual(visits, 0)

    def test_the_search_actually_visited_something(self) -> None:
        # Otherwise every assertion about the counts above is vacuous.
        self.agent.search(self.valid)
        totals = self.recorder.samples[0]["totals"]
        self.assertGreater(sum(visits for visits, _ in totals.values()), 0)

    def test_the_recorded_choice_is_the_card_that_gets_played(self) -> None:
        choice = self.agent.choose_card(self.valid)
        self.assertEqual(choice, self.recorder.samples[0]["choice"])
        self.assertNotIn(choice, self.round.player[0].hand)

    def test_the_recorded_state_is_the_one_the_student_will_see(self) -> None:
        # i.e. taken before `choose_card` removes the card from the hand. A
        # student handed the post-removal observation would be learning to name
        # a card that the state says it no longer holds.
        expected = self.round.get_state(player_id=0)
        self.agent.choose_card(self.valid)
        np.testing.assert_array_equal(expected, self.recorder.samples[0]["state"])

    def test_it_records_once_per_searched_decision(self) -> None:
        agent, recorder = counting_agent()
        make_round(agent).start()
        self.assertEqual(10, agent.decisions)  # one per card in the hand
        self.assertGreater(agent.forced, 0)  # the last trick at the very least
        self.assertEqual(agent.decisions - agent.forced, len(recorder))


class ForcedMoveTest(unittest.TestCase):
    def test_a_forced_move_records_nothing(self) -> None:
        agent, recorder = recording_agent(simulations=10**9, determinizations=10**9)
        round_ = make_round(agent)
        only = round_.player[0].hand[0]
        self.assertEqual(only, agent.search({only}))  # would hang if searched
        self.assertEqual(0, len(recorder))


class GroupingTest(unittest.TestCase):
    def test_every_sample_of_one_round_shares_a_group(self) -> None:
        agent, recorder = recording_agent()
        make_round(agent).start()
        first = {sample["group"] for sample in recorder.samples}
        self.assertEqual(1, len(first))

    def test_a_new_round_starts_a_new_group(self) -> None:
        agent, recorder = recording_agent()
        for _ in range(2):
            make_round(agent).start()
            agent.new_round_event()  # `Round.start` fires this before dealing
        self.assertEqual(2, len({sample["group"] for sample in recorder.samples}))


class TransparencyTest(unittest.TestCase):
    """Recording must not change how the teacher plays, or the corpus
    describes an agent that never sat at the table."""

    def choice_of(self, agent: MCTSAgent) -> Card:
        round_ = make_round(agent)
        assert round_.game is not None
        return agent.search(round_.player[0].valid_moves(round_.game.trick))

    def test_recording_does_not_change_which_card_is_played(self) -> None:
        plain = MCTSAgent(simulations=60, determinizations=3, seed=7)
        recording, _ = recording_agent(simulations=60, determinizations=3, seed=7)
        self.assertEqual(self.choice_of(plain), self.choice_of(recording))

    def test_the_recorded_totals_are_what_the_choice_came_from(self) -> None:
        # `best` is a pure function of the tally, so the row on disk must
        # reproduce the recorded choice exactly.
        agent, recorder = recording_agent()
        round_ = make_round(agent)
        assert round_.game is not None
        agent.search(round_.player[0].valid_moves(round_.game.trick))
        sample = recorder.samples[0]
        self.assertEqual(
            sample["choice"],
            MCTSAgent.best(list(sample["moves"]), sample["totals"]),
        )


class ChoiceVersusVisitsTest(unittest.TestCase):
    """`best` scores by mean value, `visit_targets` by visit count. They can
    disagree, which is why the corpus stores the choice instead of deriving
    it - and why `--target value` is worth having."""

    def test_the_choice_may_disagree_with_the_most_visited_move(self) -> None:
        agent, recorder = recording_agent()
        round_ = make_round(agent)
        assert round_.game is not None
        moves = sorted(
            round_.player[0].valid_moves(round_.game.trick), key=lambda c: c.np_index
        )
        rare, popular = moves[0], moves[1]
        totals = {move: [1, 0.1] for move in moves}
        totals[popular] = [99, 9.9]  # mean 0.10
        totals[rare] = [3, 2.7]  # mean 0.90

        with patch.object(MCTSAgent, "tally", return_value=totals):
            choice = agent.search(set(moves))

        self.assertEqual(rare, choice)
        recorded = recorder.samples[0]["totals"]
        self.assertEqual(popular, max(recorded, key=lambda m: recorded[m][0]))


class SeatTest(unittest.TestCase):
    def test_it_records_from_its_own_chair(self) -> None:
        # The observation is egocentric, so a sample only means anything paired
        # with the seat it was taken from - slot 0 must be the recorder's.
        agent, recorder = recording_agent()
        round_ = make_round(agent, seat=2, soloist=1)
        while round_.next_player != 2:
            round_.step()
        expected = round_.get_state(player_id=2)
        round_.step()
        self.assertEqual(1, len(recorder))
        np.testing.assert_array_equal(expected, recorder.samples[0]["state"])
        # The role bit reads from seat 2's chair: it is defending, not soloing.
        self.assertEqual(0.0, expected[38])


if __name__ == "__main__":
    unittest.main()
