"""Candidate features: kept, pressed, original skat, position and contract."""

import numpy as np

from skat.hand import FULL_HAND

FEATURE_SCHEMA = "skat-pickup-declaration-v1"
INPUT_SIZE = 32 * 3 + 3 + 6


def declaration_state(kept, pressed, original_skat, game, seat, dealer):
    kept, pressed, original_skat = tuple(kept), tuple(pressed), tuple(original_skat)
    if (
        len(kept) != 10
        or len(pressed) != 2
        or len(original_skat) != 2
        or len(set(kept + pressed)) != 12
        or not set(kept + pressed) <= set(FULL_HAND)
        or not set(original_skat) <= set(kept + pressed)
        or seat not in range(3)
        or dealer not in range(3)
        or game.declaration.hand
        or game.declaration.ouvert
    ):
        raise ValueError("invalid pickup declaration candidate")
    kind, suit = game.declaration.kind, game.declaration.suit
    if kind == "suit":
        if suit not in range(4):
            raise ValueError("invalid suit")
        contract = suit
    elif kind == "grand":
        contract = 4
    elif kind == "null":
        contract = 5
    else:
        raise ValueError("invalid contract")
    state = np.zeros(INPUT_SIZE, dtype=np.float32)
    state[[card.np_index for card in kept]] = 1
    state[[32 + card.np_index for card in pressed]] = 1
    state[[64 + card.np_index for card in original_skat]] = 1
    state[96 + (seat - dealer - 1) % 3] = 1
    state[99 + contract] = 1
    return state
