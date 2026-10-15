"""MCTS card play from a restricted PlayerView; conservative heuristic bidding."""

from random import Random

from skat.agents import Agent
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.search import (
    SKAT_SLOT,
    PlayoutState,
    infer_voids,
    uct_search,
)
from skat.agents.search.playout import copy_trick
from skat.agents.search.samplers import UniformSampler
from skat.config import MCTSConfig
from skat.hand import FULL_HAND
from skat.rules import null_overbid_substitute


class MCTSAgent(HeuristicDeclarer, Agent):
    def __init__(
        self,
        simulations=600,
        determinizations=20,
        seed=None,
        search_pool=None,
        sampler=None,
    ):
        super().__init__(seed)
        self.config = MCTSConfig(simulations, determinizations)
        self.search_pool = search_pool
        self.sampler = sampler if sampler is not None else UniformSampler()

    def choose_card(self, valid_moves):
        return self.search(valid_moves)

    def search(self, valid_moves):
        moves = sorted(valid_moves, key=lambda c: c.np_index)
        if len(moves) == 1:
            return moves[0]
        return self.best(moves, self.tally(moves))

    @staticmethod
    def best(moves, totals):
        return max(
            sorted(moves, key=lambda c: c.np_index),
            key=lambda c: totals[c][1] / totals[c][0] if totals[c][0] else -1,
        )

    def tally(self, moves):
        totals = {c: [0, 0.0] for c in moves}
        base, remainder = divmod(self.config.simulations, self.config.determinizations)
        sampler = self.sampler.prepare(self.state)
        tasks = []
        for i in range(self.config.determinizations):
            # Each world owns its sampling/playout stream; scheduling and worker
            # count cannot change subsequent worlds or the selected card.
            rng = Random(self.rng.getrandbits(64))
            world = self.determinize(rng, sampler=sampler)
            tasks.append((world, base + int(i < remainder), rng.getstate()))
        if self.search_pool is not None:
            results = self.search_pool.search(tasks)
        else:
            results = []
            for world, iterations, rng_state in tasks:
                rng = Random()
                rng.setstate(rng_state)
                results.append(uct_search(world, iterations, rng))
        for result in results:
            for c, (visits, value) in result.items():
                totals[c][0] += visits
                totals[c][1] += value
        return totals

    def determinize(self, rng=None, sampler=None):
        rng = self.rng if rng is None else rng
        sampler = self.sampler.prepare(self.state) if sampler is None else sampler
        v = self.state
        me, solo = v.seat_id, v.solo_player_id
        seen = set(v.hand) | set(v.skat)
        for t in (*v.trick_history, v.game.trick):
            seen.update(card for _, card in t.buffer)
        hands = [[] for _ in range(3)]
        hands[me] = list(v.hand)
        capacities = {}
        for seat in range(3):
            if seat == me:
                continue
            if v.exposed_hands[seat]:
                hands[seat] = list(v.exposed_hands[seat])
                seen.update(hands[seat])
            else:
                capacities[seat] = v.hand_sizes[seat]
        if not v.skat:
            capacities[SKAT_SLOT] = 2
        voids = infer_voids((*v.trick_history, v.game.trick), v.game.trick)
        deal = sampler.deal(
            [c for c in FULL_HAND if c not in seen], capacities, voids, rng
        )
        for seat in range(3):
            if seat in deal:
                hands[seat] = deal[seat]
        skat = list(v.skat) if v.skat else deal[SKAT_SLOT]
        points = list(v.points)
        points[solo] += sum(c.value for c in skat)
        original = (
            hands[solo]
            + skat
            + [
                c
                for t in (*v.trick_history, v.game.trick)
                for seat, c in t.buffer
                if seat == solo
            ]
        )
        substitute = (
            null_overbid_substitute(original, v.highest_bid)
            if v.game.declaration.kind == "null" and v.highest_bid > v.game.value
            else None
        )
        return PlayoutState(
            copy_trick(v.game.trick),
            [sorted(h, key=lambda c: c.np_index) for h in hands],
            me,
            points,
            solo,
            v.game.declaration,
            v.game.matadors(original),
            list(v.trick_counts),
            v.highest_bid,
            substitute,
        )


class RecordingMCTSAgent(MCTSAgent):
    def __init__(self, recorder, **kwargs):
        super().__init__(**kwargs)
        self.recorder = recorder

    def new_round_event(self):
        super().new_round_event()
        self.recorder.new_round()

    def search(self, valid_moves):
        moves = sorted(valid_moves, key=lambda c: c.np_index)
        if len(moves) == 1:
            return moves[0]
        totals = self.tally(moves)
        choice = self.best(moves, totals)
        self.recorder.add(self.state.observation, totals, moves, choice)
        return choice
