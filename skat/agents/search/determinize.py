from __future__ import annotations

from collections.abc import Iterable, Mapping
from random import Random
from typing import Optional

from skat.agents.search.playout import copy_trick
from skat.card import Card
from skat.trick import Trick

#: Pseudo-slot for the two cards that lie in the skat and will never be played.
#: Their points still count for the soloist, so they have to be dealt.
SKAT_SLOT = -1


def infer_voids(
    tricks: Iterable[Trick], prototype: Trick
) -> dict[int, frozenset[Card]]:
    """
    Work out which cards each seat provably cannot hold.

    Whenever a player did not serve the forced set they are void in it, so
    every card of that set is impossible for them for the rest of the round.
    Deriving the forced set from the game's own `Trick.forced_cards` — by
    replaying each trick on an empty `prototype` — means this works unchanged
    for suit games (trump vs. suit), grand and null.
    """
    voids: dict[int, set[Card]] = {0: set(), 1: set(), 2: set()}
    for trick in tricks:
        replay = copy_trick(prototype, keep_buffer=False)
        for player_id, card in trick.buffer:
            forced = replay.forced_cards
            if forced and card not in forced:
                voids[player_id] |= forced
            replay.append(player_id, card)
    return {seat: frozenset(cards) for seat, cards in voids.items()}


def deal_unseen(
    unseen: Iterable[Card],
    capacities: Mapping[int, int],
    voids: Mapping[int, frozenset[Card]],
    rng: Random,
    attempts: int = 16,
) -> dict[int, list[Card]]:
    """
    Sample one world: hand `unseen` out over the slots in `capacities`
    (`{seat_or_SKAT_SLOT: number_of_cards}`) without giving any seat a card it
    is known to be void in.

    A greedy assignment can paint itself into a corner, so retry a few times;
    if the constraints stay unsatisfiable, fall back to an unconstrained deal —
    a slightly impossible world beats having no search at all.
    """
    cards = list(unseen)
    if sum(capacities.values()) != len(cards):
        raise ValueError(
            f"cannot place {len(cards)} unseen cards in "
            f"{sum(capacities.values())} slots"
        )
    for _ in range(attempts):
        deal = _try_deal(cards, capacities, voids, rng)
        if deal is not None:
            return deal
    unconstrained = _try_deal(cards, capacities, {}, rng)
    assert unconstrained is not None  # without voids every deal succeeds
    return unconstrained


def _try_deal(
    cards: list[Card],
    capacities: Mapping[int, int],
    voids: Mapping[int, frozenset[Card]],
    rng: Random,
) -> Optional[dict[int, list[Card]]]:
    slots = list(capacities)
    remaining = dict(capacities)
    deal: dict[int, list[Card]] = {slot: [] for slot in slots}
    shuffled = list(cards)
    rng.shuffle(shuffled)
    # Most constrained card first, keeping the shuffle as the tie-break so the
    # sampling stays uniform-ish among equally constrained cards.
    shuffled.sort(key=lambda card: sum(_fits(card, slot, voids) for slot in slots))
    for card in shuffled:
        options = [
            slot for slot in slots if remaining[slot] > 0 and _fits(card, slot, voids)
        ]
        if not options:
            return None
        slot = rng.choice(options)
        deal[slot].append(card)
        remaining[slot] -= 1
    return deal


def _fits(card: Card, slot: int, voids: Mapping[int, frozenset[Card]]) -> bool:
    # The skat is dealt before anyone follows suit, so it constrains nothing.
    return slot == SKAT_SLOT or card not in voids.get(slot, frozenset())
