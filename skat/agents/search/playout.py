from __future__ import annotations

import copy
from typing import Optional

from skat.card import Card
from skat.trick import Trick


class PlayoutState:
    """
    A perfect-information model of the card-play phase, cheap enough to clone
    thousands of times per move.

    `Round` itself can't serve as the search model: it owns the `Player`s, which
    own the agents, which point back at the `Round` — cloning it would clone
    (and re-enter) every strategy, including torch models. `PlayoutState` keeps
    only what the rules need — the three hands, the trick in progress and the
    points booked so far — and delegates every rule decision to a `Trick`
    instance, so suit/grand/null behaviour stays in `skat.games` and is never
    duplicated here.
    """

    __slots__ = (
        "trick",
        "hands",
        "to_move",
        "points",
        "soloist",
        "is_null",
        "soloist_took_trick",
        "_forced_cache",
    )

    def __init__(
        self,
        trick: Trick,
        hands: list[list[Card]],
        to_move: int,
        points: list[int],
        soloist: int,
        is_null: bool = False,
        soloist_took_trick: bool = False,
        forced_cache: Optional[dict] = None,
    ) -> None:
        self.trick = trick
        self.hands = hands
        self.to_move = to_move
        self.points = points
        self.soloist = soloist
        self.is_null = is_null
        self.soloist_took_trick = soloist_took_trick
        # Shared on purpose: a pure memo of `Trick.forced_cards`, valid for the
        # whole search because every clone plays the same declared game.
        self._forced_cache: dict = {} if forced_cache is None else forced_cache

    def clone(self) -> PlayoutState:
        return PlayoutState(
            trick=copy_trick(self.trick),
            hands=[list(hand) for hand in self.hands],
            to_move=self.to_move,
            points=list(self.points),
            soloist=self.soloist,
            is_null=self.is_null,
            soloist_took_trick=self.soloist_took_trick,
            forced_cache=self._forced_cache,
        )

    @property
    def is_terminal(self) -> bool:
        return not any(self.hands)

    def legal_moves(self) -> list[Card]:
        """The cards `to_move` may play — same rule as `Player.valid_moves`."""
        hand = self.hands[self.to_move]
        forced = self._forced()
        if not forced:
            return list(hand)
        follow = [card for card in hand if card in forced]
        return follow or list(hand)

    def apply(self, card: Card) -> None:
        player = self.to_move
        self.hands[player].remove(card)
        self.trick.append(player, card)
        if not self.trick.is_full:
            self.to_move = (player + 1) % 3
            return
        winner = self.trick.winner
        assert winner is not None  # a full trick always has a winner
        self.points[winner] += self.trick.value
        if winner == self.soloist:
            self.soloist_took_trick = True
        self.to_move = winner
        self.trick = copy_trick(self.trick, keep_buffer=False)

    def rewards(self) -> tuple[float, float, float]:
        """
        Outcome of the round in `[0, 1]` per seat. Skat is a two-party game, so
        the defenders always get the complement of the soloist's value.
        """
        if self.is_null:
            # The null soloist wins by taking no trick at all (ISkO 4.2.3).
            soloist_value = 0.0 if self.soloist_took_trick else 1.0
        else:
            points = self.points[self.soloist]
            # 61 of the 120 card points decide the game (ISkO 4.1.1). The point
            # term is only a tie-breaker between moves that agree on win/loss,
            # but it keeps the signal from being all-or-nothing.
            soloist_value = (1.0 if points > 60 else 0.0) * 0.9 + points / 1200.0
        return tuple(  # type: ignore[return-value]
            soloist_value if seat == self.soloist else 1.0 - soloist_value
            for seat in range(3)
        )

    def outcome_rewards(self) -> tuple[float, float, float]:
        """Binary party win used when search estimates a win probability."""
        if self.is_null:
            soloist_won = not self.soloist_took_trick
        else:
            soloist_won = self.points[self.soloist] > 60
        soloist_value = float(soloist_won)
        return tuple(  # type: ignore[return-value]
            soloist_value if seat == self.soloist else 1.0 - soloist_value
            for seat in range(3)
        )

    def _forced(self) -> frozenset:
        """
        `Trick.forced_cards` is a (slow) pure function of the trick's lead card
        for every `Game` — it rebuilds the whole trump/suit tuple on each call.
        Memoise it by lead card, which is what the search hammers.
        """
        if not self.trick.buffer:
            return frozenset()
        lead = self.trick.buffer[0].card
        cached = self._forced_cache.get(lead)
        if cached is None:
            cached = frozenset(self.trick.forced_cards)
            self._forced_cache[lead] = cached
        return cached


def copy_trick(trick: Trick, keep_buffer: bool = True) -> Trick:
    """
    Copy a `Trick` without knowing its concrete type or constructor arguments
    (`SuitGameTrick` takes a suit, `GrandTrick`/`NullTrick` take nothing). A
    shallow copy carries the rule configuration over; only the buffer needs to
    become a fresh list so clones don't share it.
    """
    clone = copy.copy(trick)
    clone.buffer = list(trick.buffer) if keep_buffer else []
    return clone
