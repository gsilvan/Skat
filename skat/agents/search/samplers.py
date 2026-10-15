"""Interchangeable complete-world samplers for MCTS."""

from skat.agents.search.determinize import WeightedDealer, deal_unseen


class UniformSampler:
    def prepare(self, view):
        return self

    def deal(self, unseen, capacities, voids, rng):
        return deal_unseen(unseen, capacities, voids, rng)


class PreparedWeightedSampler:
    def __init__(self, weights):
        self.weights = weights
        self.dealer = None

    def deal(self, unseen, capacities, voids, rng):
        if self.dealer is None:
            self.dealer = WeightedDealer(unseen, capacities, voids, self.weights)
        return self.dealer.sample(rng)
