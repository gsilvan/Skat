import numpy as np

from skat.card import RANKS, SUITS, Card


def to_numpy(hand: list[Card]):
    arr = np.zeros(32, dtype=bool)
    i = 0
    for j, _ in enumerate(SUITS):
        for k, _ in enumerate(RANKS):
            if Card(j, k) in hand:
                arr[i] = True
            i += 1
    return arr


def disjoint(items):
    """Return True if the flatten set of list items is disjoint."""
    union = set()
    for item in items:
        for x in item:
            if x in union:
                return False
            union.add(x)
    return True
