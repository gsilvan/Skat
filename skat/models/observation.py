"""The shape of what `Round.get_state` produces.

This lives on its own because it belongs to the observation, not to any one
network. Four models and half a dozen tests need to know how wide the
observation is, and for a long time they all reached through `SuitSoloNet` -
the DQN's Q-network - to find out, which is why test docstrings ended up
talking about "SuitSoloNet.INPUT_SIZE" when what they meant was "how many
numbers `get_state` returns".

`Round.get_state` and the widths below are coupled by hand. Nothing enforces
it at import time; `tests/test_state_vector.py` is what asserts the total
matches what the engine actually builds.

Note what is deliberately *not* here: the per-block slice offsets. Both
`tests/test_state_vector.py` and `skat.models.card_transformer` restate those
independently, and that is the point - two separate statements of a layout are
what catch a change to one of them. Sharing a single constant would make them
agree by construction and pin nothing.
"""

#: Own cards, 32-hot.
HAND = 32
#: Trump suit one-hot, then is_grand, is_null.
GAME = 6
#: Am I the soloist.
ROLE = 1
#: Where the soloist sits, relative to me.
SOLOIST_SEAT = 3
#: My party's card points, then theirs.
POINTS = 2
#: 3 x 32, completed tricks, egocentric - slot 0 is me.
PLAYED_CARDS = 96
#: 3 x 32, the live trick, same seat order.
CURRENT_TRICK = 96
#: Front / middle / back hand, my own position.
FRONT_HAND = 3
#: Card points on the table, scaled.
TRICK_VALUE = 1

#: Listed in the order `Round.get_state` concatenates them.
BLOCKS = (
    HAND,
    GAME,
    ROLE,
    SOLOIST_SEAT,
    POINTS,
    PLAYED_CARDS,
    CURRENT_TRICK,
    FRONT_HAND,
    TRICK_VALUE,
)

#: How many numbers `Round.get_state` returns.
INPUT_SIZE = sum(BLOCKS)

#: One output per card in the pack, whatever the network means by it - a
#: Q-value, a policy logit, a location distribution.
OUTPUT_SIZE = 32
