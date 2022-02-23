import numpy as np

from skat.card import Card, RANKS, SUITS


def to_numpy(hand: list[Card]):
    arr = np.zeros(32, dtype=bool)
    i = 0
    for j, _ in enumerate(SUITS):
        for k, _ in enumerate(RANKS):
            if Card(j, k) in hand:
                arr[i] = True
            i += 1
    return arr
