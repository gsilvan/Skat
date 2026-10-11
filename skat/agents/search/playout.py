"""Cheap complete-world card play with the engine's official scoring."""

import copy

from skat.rules import evaluate_game


def copy_trick(trick, keep_buffer=True):
    clone = copy.copy(trick)
    clone.buffer = list(trick.buffer) if keep_buffer else []
    return clone


class PlayoutState:
    def __init__(
        self,
        trick,
        hands,
        to_move,
        points,
        soloist,
        declaration,
        matadors,
        trick_counts=None,
        highest_bid=0,
        null_substitute=None,
    ):
        self.trick = trick
        self.hands = hands
        self.to_move = to_move
        self.points = points
        self.soloist = soloist
        self.declaration = declaration
        self.matadors = matadors
        self.trick_counts = trick_counts or [0, 0, 0]
        self.highest_bid = highest_bid
        self.null_substitute = null_substitute

    def clone(self):
        return PlayoutState(
            copy_trick(self.trick),
            [list(h) for h in self.hands],
            self.to_move,
            list(self.points),
            self.soloist,
            self.declaration,
            self.matadors,
            list(self.trick_counts),
            self.highest_bid,
            self.null_substitute,
        )

    @property
    def is_terminal(self):
        return not any(self.hands) or (
            self.declaration.kind == "null" and self.trick_counts[self.soloist] > 0
        )

    def legal_moves(self):
        hand = self.hands[self.to_move]
        forced = self.trick.forced_cards
        return [c for c in hand if c in forced] or list(hand)

    def apply(self, card):
        player = self.to_move
        self.hands[player].remove(card)
        self.trick.append(player, card)
        if not self.trick.is_full:
            self.to_move = (player + 1) % 3
            return
        winner = self.trick.winner
        self.points[winner] += self.trick.value
        self.trick_counts[winner] += 1
        self.to_move = winner
        self.trick = copy_trick(self.trick, keep_buffer=False)

    def result(self):
        if not self.is_terminal:
            raise ValueError("search result requires a terminal state")
        extra = {}
        if self.null_substitute is not None:
            extra = dict(
                null_overbid_value=self.null_substitute[0],
                null_overbid_declaration=self.null_substitute[1],
            )
        return evaluate_game(
            self.declaration,
            self.matadors,
            (self.points[self.soloist], sum(self.points) - self.points[self.soloist]),
            (
                self.trick_counts[self.soloist],
                sum(self.trick_counts) - self.trick_counts[self.soloist],
            ),
            self.highest_bid,
            soloist=self.soloist,
            **extra,
        )

    def rewards(self):
        won = self.result().soloist_won
        return tuple(float((seat == self.soloist) == won) for seat in range(3))
