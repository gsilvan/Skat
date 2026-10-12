"""Only the original ten cards and fore/middle/rear position are observable."""

import numpy as np

from skat.hand import FULL_HAND

FEATURE_SCHEMA = "skat-original-hand-position-v1"
INPUT_SIZE = 35


def bidding_state(cards, seat, dealer):
    cards = tuple(cards)
    if (
        len(cards) != 10
        or len(set(cards)) != 10
        or not set(cards) <= set(FULL_HAND)
        or seat not in range(3)
        or dealer not in range(3)
    ):
        raise ValueError("bidding requires ten distinct cards and valid seats")
    state = np.zeros(INPUT_SIZE, dtype=np.float32)
    state[[card.np_index for card in cards]] = 1
    state[32 + (seat - dealer - 1) % 3] = 1
    return state
