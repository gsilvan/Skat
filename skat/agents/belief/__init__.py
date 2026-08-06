"""Estimating where the cards you cannot see actually are.

`MCTSAgent` deals with imperfect information by sampling complete deals
consistent with what it has seen, and it samples them **uniformly**: every
consistent world is equally likely. That is a real cost. A soloist who declared
Grand almost certainly holds jacks, so half the sampled worlds are ones nobody
would ever reach, and the simulations spent on them are wasted.

This package trains a network to say where each unseen card probably is, and
`skat.agents.belief_mcts.BeliefMCTSAgent` samples its determinizations from
that instead. The baseline it has to beat is not a flat guess but the
capacity-proportional prior the uniform dealer already implies - see
`loss.capacity_prior`.

Ground truth is free: the engine knows every hand at every instant, so a
self-play run labels itself. That is also the *right* source rather than merely
the cheap one - beliefs are always relative to an opponent model, and the model
the search will face is our own agents, not the humans in the ISS archives.

Same layering rule as `skat.agents.distill`, `search` and `rl`: nothing in here
imports `Agent`, `Player` or `Round`. The two pieces that must know about them
- the recorder and the agent - live one level up, in
`skat.agents.belief_recorder` and `skat.agents.belief_mcts`.
"""

from skat.agents.belief.checkpoint import BELIEF_FORMAT, load_belief, save_belief
from skat.agents.belief.loss import (
    accuracy_of,
    belief_loss,
    belief_probabilities,
    capacity_prior,
    cross_entropy_of,
)
from skat.agents.belief.samples import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_SHARD_SIZE,
    KNOWN,
    SHARD_VERSION,
    BeliefSamples,
    BeliefShardWriter,
    load_belief_shards,
)

__all__ = [
    "BELIEF_FORMAT",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_SHARD_SIZE",
    "KNOWN",
    "SHARD_VERSION",
    "BeliefSamples",
    "BeliefShardWriter",
    "accuracy_of",
    "belief_loss",
    "belief_probabilities",
    "capacity_prior",
    "cross_entropy_of",
    "load_belief",
    "load_belief_shards",
    "save_belief",
]
