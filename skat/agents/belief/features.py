"""Only information available in a detached PlayerView enters belief inputs."""

import numpy as np

from skat.agents.search.determinize import SKAT_SLOT, infer_voids
from skat.hand import FULL_HAND
from skat.models.observation import INPUT_SIZE as OBSERVATION_SIZE

SCHEMA = "skat-belief-history-v1"
INPUT_SIZE = OBSERVATION_SIZE + 64 + 64
KNOWN = -1


def belief_input(view):
    """Observation, two opponent void masks, and public trick timing per card."""
    result = np.zeros(INPUT_SIZE, dtype=np.float32)
    result[:OBSERVATION_SIZE] = view.observation
    voids = infer_voids((*view.trick_history, view.game.trick), view.game.trick)
    for offset, seat in enumerate(((view.seat_id + 1) % 3, (view.seat_id + 2) % 3)):
        result[
            OBSERVATION_SIZE
            + offset * 32
            + np.array([c.np_index for c in voids[seat]], dtype=np.intp)
        ] = 1
    timing = result[OBSERVATION_SIZE + 64 :].reshape(2, 32)
    for trick_index, trick in enumerate((*view.trick_history, view.game.trick)):
        for order, (_, card) in enumerate(trick.buffer):
            timing[0, card.np_index] = (trick_index + 1) / 10
            timing[1, card.np_index] = (order + 1) / 3
    return result


def locations(view, hands, skat):
    """Ground truth for unseen cards; exposed cards carry no prediction target."""
    known = set(view.hand) | set(view.skat)
    for trick in (*view.trick_history, view.game.trick):
        known.update(card for _, card in trick.buffer)
    for hand in view.exposed_hands:
        known.update(hand)
    result = np.full(32, KNOWN, dtype=np.int8)
    slots = ((view.seat_id + 1) % 3, (view.seat_id + 2) % 3)
    for location, seat in enumerate(slots):
        for card in hands[seat]:
            if card not in known:
                result[card.np_index] = location
    for card in skat:
        if card not in known:
            result[card.np_index] = 2
    if any(result[c.np_index] == KNOWN for c in FULL_HAND if c not in known):
        raise ValueError("unseen card has no ground-truth location")
    return result


def hidden_deal(view, hands, skat):
    """Absolute-seat labelled world for held-out likelihood measurements."""
    result = {
        (view.seat_id + 1) % 3: tuple(hands[(view.seat_id + 1) % 3]),
        (view.seat_id + 2) % 3: tuple(hands[(view.seat_id + 2) % 3]),
        SKAT_SLOT: tuple(skat),
    }
    for seat, hand in enumerate(view.exposed_hands):
        if hand:
            result.pop(seat, None)
    if view.skat:
        result.pop(SKAT_SLOT)
    return result
