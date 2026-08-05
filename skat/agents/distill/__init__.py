"""Distilling `skat.agents.mcts.MCTSAgent` into a policy network.

MCTS is far stronger than the DQN but pays `simulations` playouts for every
card. This package turns its searches into a supervised corpus - the
observation it looked at, the visit counts and values its trees came back with,
and the card it played - and trains a network to predict that in one forward
pass. `skat.agents.policy.PolicyAgent` is what plays it.

Kept separate from the agents the same way `skat.agents.search` and
`skat.agents.rl` are: nothing in here knows about `Agent`, `Player` or `Round`,
it deals in arrays, tensors and cards. `RecordingMCTSAgent` lives next to the
agent it subclasses instead, in `skat.agents.mcts`.

This is stage 1 and 2 of expert iteration. Stage 3 - feeding these logits back
into `skat.agents.search.uct` as a PUCT prior, so the search itself gets
stronger per simulation instead of merely cheaper - is where the policy stops
being a copy of its teacher. It is not built yet.
"""

from skat.agents.distill.checkpoint import POLICY_FORMAT, load_policy, save_policy
from skat.agents.distill.loss import masked_log_softmax, policy_loss
from skat.agents.distill.samples import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_SHARD_SIZE,
    SHARD_VERSION,
    Samples,
    ShardWriter,
    load_shards,
)
from skat.agents.distill.targets import value_targets, visit_targets

__all__ = [
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_SHARD_SIZE",
    "POLICY_FORMAT",
    "SHARD_VERSION",
    "Samples",
    "ShardWriter",
    "load_policy",
    "load_shards",
    "masked_log_softmax",
    "policy_loss",
    "save_policy",
    "value_targets",
    "visit_targets",
]
