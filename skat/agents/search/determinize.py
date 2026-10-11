"""Infer public voids and sample only worlds consistent with that evidence."""

from functools import lru_cache

from skat.agents.search.playout import copy_trick

SKAT_SLOT = -1


def infer_voids(tricks, prototype):
    voids = {s: set() for s in range(3)}
    for trick in tricks:
        replay = copy_trick(prototype, keep_buffer=False)
        for seat, card in trick.buffer:
            forced = replay.forced_cards
            if forced and card not in forced:
                voids[seat].update(forced)
            replay.append(seat, card)
    return {s: frozenset(v) for s, v in voids.items()}


@lru_cache(maxsize=50000)
def _completions(allowed, remaining):
    if not allowed:
        return int(not any(remaining))
    total = 0
    for slot in allowed[0]:
        if remaining[slot]:
            capacities = list(remaining)
            capacities[slot] -= 1
            total += _completions(allowed[1:], tuple(capacities))
    return total


def deal_unseen(unseen, capacities, voids, rng):
    """Uniform distribution over complete feasible assignments.

    Completion counts include slot capacity, so the two skat slots do not
    receive the same per-card probability as an opponent's ten-card hand.
    Impossible constraints raise instead of silently discarding observed voids.
    """
    cards = sorted(unseen, key=lambda c: c.np_index)
    slots = sorted(capacities)
    if any(type(n) is not int or n < 0 for n in capacities.values()) or sum(
        capacities.values()
    ) != len(cards):
        raise ValueError("unseen cards do not match remaining capacities")
    deal = {slot: [] for slot in slots}
    options = {
        c: tuple(
            i
            for i, s in enumerate(slots)
            if s == SKAT_SLOT or c not in voids.get(s, ())
        )
        for c in cards
    }
    if all(len(options[c]) == len(slots) for c in cards):
        rng.shuffle(cards)
        offset = 0
        for slot in slots:
            n = capacities[slot]
            deal[slot] = cards[offset : offset + n]
            offset += n
        return deal
    cards.sort(key=lambda c: len(options[c]))
    allowed = tuple(options[c] for c in cards)
    remaining = tuple(capacities[s] for s in slots)
    if not _completions(allowed, remaining):
        raise ValueError("public voids and capacities admit no complete world")
    for card in cards:
        choices = []
        for i in allowed[0]:
            if remaining[i]:
                next_remaining = list(remaining)
                next_remaining[i] -= 1
                count = _completions(allowed[1:], tuple(next_remaining))
                if count:
                    choices.append((i, tuple(next_remaining), count))
        draw = rng.randrange(sum(count for _, _, count in choices))
        for i, next_remaining, count in choices:
            if draw < count:
                deal[slots[i]].append(card)
                remaining = next_remaining
                break
            draw -= count
        allowed = allowed[1:]
    return deal
