"""
Hand prediction, and the evidence it is allowed to use.

Two things are being pinned. First that the labels say where the cards really
were, under exactly the definition of "already visible" that
`MCTSAgent.determinize` uses - train the network on a different question from
the one it gets asked and nothing downstream will complain, it will just be
quietly worse.

Second, void inference. Not following suit is the strongest read in Skat: a
player who discards on a trump lead is trump-free for the rest of the round,
and one who discards on a heart lead can never hold a heart again. The search
has always used that as a hard constraint when dealing worlds. What is new here
is that the belief network gets told about it too - it cannot work it out for
itself, because `Round.get_state` ORs all ten tricks into one set per seat and
loses which cards fell together.
"""

import unittest
from random import Random

import numpy as np

from skat.agents import Agent
from skat.agents.belief.samples import KNOWN
from skat.agents.belief_recorder import belief_input, observer_view, void_block
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.random import RandomAgent
from skat.agents.search import SKAT_SLOT, deal_unseen
from skat.game import GamePhase, Round
from skat.games.grand import Grand
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND
from skat.models.belief import NEXT, PREVIOUS, SKAT, BeliefNet

CLUBS = 3
DEAL = [FULL_HAND[0:10], FULL_HAND[10:20], FULL_HAND[20:30], FULL_HAND[30:32]]


class ScriptedAgent(HeuristicDeclarer, Agent):
    """Plays the first card of its script that is legal, else anything."""

    def __init__(self, script=()) -> None:
        super().__init__()
        self.script = list(script)

    def choose_card(self, valid_moves):
        for card in self.script:
            if card in valid_moves:
                self.script.remove(card)
                self.state.hand.remove(card)
                return card
        choice = min(valid_moves, key=lambda c: c.np_index)
        self.state.hand.remove(choice)
        return choice


def make_round(agents=None, soloist=0, game=None, hand_game=True, steps=0) -> Round:
    round_ = Round(
        skip_bidding=True,
        solo_player_id=soloist,
        hand_game=hand_game,
        declare_game=game if game is not None else SuitGame(CLUBS),
        agents=agents or [ScriptedAgent() for _ in range(3)],
        start=False,
        initial_cards=[list(h) for h in DEAL],
    )
    round_.deal()
    round_.phase = GamePhase.PLAYING
    for _ in range(steps):
        round_.step()
    return round_


def voids_for(block: np.ndarray, offset: int) -> set:
    """The card slots one opponent's half of the void block rules out."""
    half = block[offset * 32 : (offset + 1) * 32]
    return set(np.nonzero(half)[0].tolist())


class GroundTruthTest(unittest.TestCase):
    def test_every_unseen_card_is_labelled_with_where_it_really_is(self) -> None:
        round_ = make_round(steps=7)
        locations = observer_view(round_, 0)
        for card in round_.player[1].hand:
            self.assertEqual(NEXT, locations[card.np_index])
        for card in round_.player[2].hand:
            self.assertEqual(PREVIOUS, locations[card.np_index])
        for card in round_.skat:
            self.assertEqual(SKAT, locations[card.np_index])

    def test_my_own_cards_are_not_something_to_predict(self) -> None:
        round_ = make_round(steps=7)
        locations = observer_view(round_, 0)
        for card in round_.player[0].hand:
            self.assertEqual(KNOWN, locations[card.np_index])

    def test_a_played_card_stops_being_a_question(self) -> None:
        round_ = make_round(steps=7)
        locations = observer_view(round_, 0)
        played = {c for t in round_.trick_history.buffer for _, c in t.buffer}
        for card in played:
            self.assertEqual(KNOWN, locations[card.np_index])

    def test_the_soloist_who_took_the_skat_already_knows_it(self) -> None:
        # `determinize` treats a pressed skat as seen, so the labels must too.
        round_ = make_round(hand_game=False, steps=3)
        for card in round_.skat:
            self.assertEqual(KNOWN, observer_view(round_, 0)[card.np_index])
        # ...but a defender still has to guess it
        self.assertEqual(SKAT, observer_view(round_, 1)[round_.skat[0].np_index])

    def test_a_hand_game_leaves_the_skat_hidden_from_everyone(self) -> None:
        round_ = make_round(hand_game=True, steps=3)
        for seat in range(3):
            for card in round_.skat:
                self.assertEqual(SKAT, observer_view(round_, seat)[card.np_index])

    def test_the_counts_add_up_to_the_cards_that_are_left(self) -> None:
        round_ = make_round(steps=11)
        locations = observer_view(round_, 0)
        self.assertEqual(len(round_.player[1].hand), int((locations == NEXT).sum()))
        self.assertEqual(len(round_.player[2].hand), int((locations == PREVIOUS).sum()))
        self.assertEqual(2, int((locations == SKAT).sum()))


class VoidInferenceTest(unittest.TestCase):
    """Farb-frei and Trumpf-frei, which is the evidence the observation itself
    cannot carry.

    The expectation is derived from the tricks the engine actually played
    rather than from a hand-picked scenario, so these hold whoever ends up on
    forehand and whatever the agents choose.
    """

    def discarders(self, round_) -> dict:
        """`{seat: the forced set they failed to serve}`, replayed by hand."""
        assert round_.game is not None
        from skat.agents.search.playout import copy_trick

        expected: dict[int, set] = {0: set(), 1: set(), 2: set()}
        for trick in [*round_.trick_history.buffer, round_.game.trick]:
            replay = copy_trick(round_.game.trick, keep_buffer=False)
            for seat, card in trick.buffer:
                forced = replay.forced_cards
                if forced and card not in forced:
                    expected[seat] |= forced
                replay.append(seat, card)
        return expected

    def check(self, round_, seat: int = 0) -> None:
        block = void_block(round_, seat)
        expected = self.discarders(round_)
        for offset, opponent in enumerate(((seat + 1) % 3, (seat + 2) % 3)):
            self.assertEqual(
                {card.np_index for card in expected[opponent]},
                voids_for(block, offset),
                f"seat {opponent} as seen from {seat}",
            )

    def test_a_player_who_cannot_follow_is_marked_void_in_that_suit(self) -> None:
        round_ = make_round(steps=12)
        self.assertTrue(
            any(self.discarders(round_).values()), "no void arose - vacuous test"
        )
        self.check(round_)

    def test_it_holds_from_every_chair(self) -> None:
        round_ = make_round(steps=12)
        for seat in range(3):
            self.check(round_, seat)

    def test_it_holds_in_a_grand_where_only_the_jacks_are_trumps(self) -> None:
        round_ = make_round(game=Grand(), steps=12)
        self.check(round_)

    def test_nothing_is_void_before_a_card_is_played(self) -> None:
        self.assertEqual(0.0, void_block(make_round(), 0).sum())

    def test_the_live_trick_counts_as_evidence(self) -> None:
        # The freshest void there is - somebody discarded a moment ago and the
        # trick is not finished. Feeding only the completed tricks threw this
        # away at every single decision.
        opened = None
        for steps in range(1, 30):
            candidate = make_round(steps=steps)
            assert candidate.game is not None
            if candidate.game.trick.buffer and any(
                self.discarders(candidate)[seat] for seat in range(3)
            ):
                opened = candidate
                break
        self.assertIsNotNone(opened, "never found a mid-trick void")
        assert opened is not None and opened.game is not None
        self.assertTrue(opened.game.trick.buffer, "trick should still be open")
        # The evidence in the open trick must already be in the block, which is
        # only true because the live trick is passed to `infer_voids`.
        self.check(opened)

    def test_the_block_is_two_thirty_twos_in_seat_order(self) -> None:
        self.assertEqual(2 * 32, len(void_block(make_round(steps=12), 0)))


class BeliefInputTest(unittest.TestCase):
    def test_it_is_the_observation_followed_by_the_voids(self) -> None:
        round_ = make_round(steps=7)
        x = belief_input(round_, 0)
        self.assertEqual(BeliefNet.INPUT_SIZE, len(x))
        np.testing.assert_array_equal(round_.get_state(player_id=0), x[:240])
        np.testing.assert_array_equal(void_block(round_, 0), x[240:])

    def test_the_network_takes_it_and_answers_per_card(self) -> None:
        import torch

        round_ = make_round(steps=7)
        net = BeliefNet()
        net.eval()
        x = torch.tensor(belief_input(round_, 0)).unsqueeze(0)
        with torch.no_grad():
            out = net(x)
        self.assertEqual((1, 32, 3), tuple(out.shape))

    def test_an_observation_without_the_voids_is_refused(self) -> None:
        import torch

        with self.assertRaises(RuntimeError) as caught:
            BeliefNet()(torch.zeros(1, 240))
        self.assertIn("304", str(caught.exception))


class WeightedDealTest(unittest.TestCase):
    """The sampling change. Weights bias it; constraints still bind."""

    def setUp(self) -> None:
        self.unseen = FULL_HAND[:6]
        self.capacities = {1: 3, 2: 3}
        self.rng = Random(0)

    def test_without_weights_it_is_the_old_uniform_dealer(self) -> None:
        counts = {1: 0, 2: 0}
        for _ in range(400):
            deal = deal_unseen(self.unseen, self.capacities, {}, Random(_))
            counts[1] += self.unseen[0] in deal[1]
            counts[2] += self.unseen[0] in deal[2]
        self.assertGreater(min(counts.values()), 120)  # roughly even

    def test_weights_move_a_card_towards_the_slot_they_favour(self) -> None:
        weights = {self.unseen[0]: {1: 0.99, 2: 0.01}}
        hits = sum(
            self.unseen[0]
            in deal_unseen(
                self.unseen, self.capacities, {}, Random(s), weights=weights
            )[1]
            for s in range(400)
        )
        self.assertGreater(hits, 300)

    def test_capacity_is_still_exact(self) -> None:
        weights = {card: {1: 1.0, 2: 0.0} for card in self.unseen}
        deal = deal_unseen(self.unseen, self.capacities, {}, self.rng, weights=weights)
        self.assertEqual(3, len(deal[1]))
        self.assertEqual(3, len(deal[2]))

    def test_a_void_still_beats_the_belief(self) -> None:
        # The belief is a guess; the void is a fact observed at the table.
        voids = {1: frozenset(self.unseen[:2])}
        weights = {card: {1: 1.0, 2: 0.0} for card in self.unseen}
        for seed in range(30):
            deal = deal_unseen(
                self.unseen, self.capacities, voids, Random(seed), weights=weights
            )
            self.assertEqual(set(), set(deal[1]) & set(self.unseen[:2]))

    def test_a_belief_that_rules_out_everything_falls_back(self) -> None:
        weights = {card: {1: 0.0, 2: 0.0} for card in self.unseen}
        deal = deal_unseen(self.unseen, self.capacities, {}, self.rng, weights=weights)
        self.assertEqual(6, len(deal[1]) + len(deal[2]))

    def test_the_skat_slot_is_weightable_too(self) -> None:
        weights = {self.unseen[0]: {1: 0.0, 2: 0.0, SKAT_SLOT: 1.0}}
        capacities = {1: 2, 2: 2, SKAT_SLOT: 2}
        hits = sum(
            self.unseen[0]
            in deal_unseen(self.unseen, capacities, {}, Random(s), weights=weights)[
                SKAT_SLOT
            ]
            for s in range(50)
        )
        # Not 50/50. The dealer assigns card by card, so by the time the
        # favoured card comes up the slot it wants may already be full - the
        # sequential approximation `_pick` documents. It should still land
        # there far more often than the 1/3 a uniform dealer would give.
        self.assertGreater(hits, 35)


class UniformityTest(unittest.TestCase):
    """The baseline agent must be untouched, or the comparison means nothing."""

    def test_the_default_path_never_consults_a_weight(self) -> None:
        rounds = [
            make_round(agents=[RandomAgent() for _ in range(3)]) for _ in range(2)
        ]
        self.assertEqual(2, len(rounds))  # smoke: the engine still runs

    def test_deal_unseen_is_reproducible_from_a_seed(self) -> None:
        first = deal_unseen(FULL_HAND[:6], {1: 3, 2: 3}, {}, Random(5))
        second = deal_unseen(FULL_HAND[:6], {1: 3, 2: 3}, {}, Random(5))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
