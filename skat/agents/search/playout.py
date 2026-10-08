from __future__ import annotations

import copy
from typing import Optional

from skat.card import Card
from skat.rewards import victory_bonus, victory_status
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
        "secured_winner",
        "overbid",
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
        secured_winner: Optional[int] = None,
        overbid: bool = False,
    ) -> None:
        self.trick = trick
        self.hands = hands
        self.to_move = to_move
        self.points = points
        self.soloist = soloist
        self.is_null = is_null
        self.soloist_took_trick = soloist_took_trick
        self.overbid = overbid
        # Shared on purpose: a pure memo of `Trick.forced_cards`, valid for the
        # whole search because every clone plays the same declared game.
        self._forced_cache: dict = {} if forced_cache is None else forced_cache
        # Frozen at the search root, never switched on by a rollout. Otherwise
        # future bonuses would already affect choices before victory is secure.
        self.secured_winner = (
            victory_status(points[soloist], sum(points) - points[soloist], overbid)
            if secured_winner is None and not is_null
            else (secured_winner or 0)
        )

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
            secured_winner=self.secured_winner,
            overbid=self.overbid,
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
            soloist_value = 0.0 if self.overbid or self.soloist_took_trick else 1.0
        else:
            points = self.points[self.soloist]
            if self.secured_winner == 1:
                soloist_value = victory_bonus(points, True)
            elif self.secured_winner == -1:
                soloist_value = 1.0 - victory_bonus(sum(self.points) - points, False)
            else:
                soloist_value = float(not self.overbid and points > 60)
        return tuple(  # type: ignore[return-value]
            soloist_value if seat == self.soloist else 1.0 - soloist_value
            for seat in range(3)
        )

    def outcome_rewards(self) -> tuple[float, float, float]:
        """Binary party win used when search estimates a win probability."""
        if self.is_null:
            soloist_won = not self.overbid and not self.soloist_took_trick
        else:
            soloist_won = not self.overbid and self.points[self.soloist] > 60
        soloist_value = float(soloist_won)
        return tuple(  # type: ignore[return-value]
            soloist_value if seat == self.soloist else 1.0 - soloist_value
            for seat in range(3)
        )

    def bonus_rewards(self) -> tuple[float, float, float]:
        """Winning-party bonuses, separately from the primary search value."""
        if self.is_null:
            return (0.0, 0.0, 0.0)
        won = not self.overbid and self.points[self.soloist] > 60
        solo = victory_bonus(self.points[self.soloist], True) if won else 0.0
        defence = (
            victory_bonus(sum(self.points) - self.points[self.soloist], False)
            if not won
            else 0.0
        )
        return tuple(solo if seat == self.soloist else defence for seat in range(3))

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
