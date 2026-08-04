"""
Non-interactive defaults for the phase between winning the auction and playing
the first card: take the skat or not, which two cards to press, which game to
declare.

Every agent that is not a human needs these, and none of it depends on how the
agent picks *cards* — it is all hand evaluation from `skat.agents.bidding`. So
it lives here as a mixin rather than in any one agent, which is what let
`DQNAgent` stop inheriting `CommandLineAgent`'s `input()` prompts and become
trainable unattended.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

from skat.agents.bidding import aces, best_game, covering_game, trump_count
from skat.card import Card
from skat.games.grand import Grand

if TYPE_CHECKING:
    from skat.games import Game
    from skat.player import Player


class HeuristicDeclarer:
    """
    Mix in before `Agent` to get automatic skat handling.

    `declare_game` and `press_skat` deliberately weigh up the *same* twelve
    cards, so the game the skat was pressed for is the game that gets declared.
    """

    #: Supplied by `Agent` - this mixin only makes sense in front of one.
    state: Optional[Player]

    def intended_game(self) -> Game:
        """The game we expect to declare, used to value the hand while bidding."""
        assert self.state is not None
        return best_game(set(self.state.hand))

    def pickup_skat(self, state) -> bool:
        assert self.state is not None
        cards = set(self.state.hand)
        game = best_game(cards)
        # Two extra cards beat the hand-game multiplier unless the hand already
        # plays itself. Grand can never reach seven trumps - only four exist -
        # so there it is the aces that have to carry the game.
        if isinstance(game, Grand):
            strong = trump_count(cards, game) >= 3 and aces(cards) >= 3
        else:
            strong = trump_count(cards, game) >= 7
        return not strong

    def declaration(self, cards: set[Card]) -> Game:
        """
        The game to declare given all twelve cards. Has to cover what we bid:
        re-deriving the natural game after the skat arrives can land on a
        cheaper suit and turn a bid we could afford into an overbid.
        """
        round_ = self.state.public_state if self.state else None
        return covering_game(cards, round_.highest_bid if round_ else 0)

    def press_skat(self) -> list[Card]:
        assert self.state is not None
        hand = self.state.hand
        game = self.declaration(set(hand))
        trumps = set(game.trump_cards(game.suit))
        candidates = [card for card in hand if card not in trumps] or list(hand)
        lengths = Counter(card.suit for card in candidates)
        # Never an ace if avoidable; then prefer the shortest suit (a void lets
        # us trump in), and inside it the fattest card — its points are safe in
        # the skat rather than at risk on the table.
        candidates.sort(key=lambda c: (c.rank == "A", lengths[c.suit], -c.value))
        pressed = candidates[:2]
        for card in pressed:
            hand.remove(card)
        return pressed

    def declare_game(self, state) -> Game:
        assert self.state is not None
        round_ = self.state.public_state
        if round_ is None or round_.hand_game:
            # We never saw the skat, so the only declaration we can still stand
            # behind is the one the bid was priced on.
            return self.intended_game()
        # Same twelve cards `press_skat` weighed up, so the two agree.
        return self.declaration(set(self.state.hand) | set(round_.skat))
