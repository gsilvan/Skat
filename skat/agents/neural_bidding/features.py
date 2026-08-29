from __future__ import annotations

import numpy as np

from skat.card import Card
from skat.hand import Hand
from skat.models.bidding import BiddingNet


def relative_position(seat: int, dealer: int) -> int:
    """Return 0/1/2 for front/middle/rear hand before the first trick."""
    front = (dealer + 1) % 3
    return (seat - front) % 3


def bidding_state(cards: list[Card] | set[Card], seat: int, dealer: int) -> np.ndarray:
    """Encode ten cards and who leads the opening trick into 35 floats."""
    if len(cards) != 10:
        raise ValueError(f"bidding needs the original ten-card hand, got {len(cards)}")
    position = np.zeros(BiddingNet.POSITIONS, dtype=np.float32)
    position[relative_position(seat, dealer)] = 1.0
    state = np.concatenate(
        (Hand(cards).as_vector.astype(np.float32), position), dtype=np.float32
    )
    assert state.shape == (BiddingNet.INPUT_SIZE,)
    return state
