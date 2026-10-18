"""Information available to the soloist before deciding to pick up."""

import numpy as np

from skat.hand import FULL_HAND

FEATURE_SCHEMA = "skat-original-hand-bid-position-v1"
INPUT_SIZE = 36


def pickup_state(cards, bid, seat, dealer):
    cards = tuple(cards)
    if (
        len(cards) != 10
        or len(set(cards)) != 10
        or not set(cards) <= set(FULL_HAND)
        or type(bid) is not int
        or bid < 0
        or bid >= 500
        or seat not in range(3)
        or dealer not in range(3)
    ):
        raise ValueError("pickup requires ten original cards, bid and valid seats")
    state = np.zeros(INPUT_SIZE, dtype=np.float32)
    state[[card.np_index for card in cards]] = 1
    state[32 + (seat - dealer - 1) % 3] = 1
    state[35] = bid / 500
    return state
