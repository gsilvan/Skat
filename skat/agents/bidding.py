"""
Bidding behaviour, shared by every agent.

An agent's whole bidding character reduces to one number: the highest value it
is willing to say. The Sagen/Hören mechanics on top of it are identical for
everyone and live in `Agent.bid`. `BiddingStyle` is how that ceiling is
derived, and the hand evaluation it rests on lives here too.
"""

from __future__ import annotations

import bisect
import random
from enum import Enum
from typing import TYPE_CHECKING, Optional

from skat.card import SUITS, Card
from skat.games.grand import Grand
from skat.games.suit import SuitGame

if TYPE_CHECKING:
    from skat.games import Game

#: The official bid ladder (ISkO 3.3.4). Ascending, starting at "pass".
VALID_BIDS = [
    0, 18, 20, 22, 24, 27, 30, 33, 36, 40, 44, 45, 48, 50, 54, 55, 60, 63, 66,
    70, 72, 77, 80, 81, 84, 88, 90, 96, 99, 100, 108, 110, 117, 120, 121, 126,
    130, 132, 135, 140, 143, 144, 150, 153, 154, 156, 160, 162, 165, 168, 170,
    171, 176, 180, 187, 189, 190, 192, 198, 200, 204, 207, 209, 210, 216, 220,
    225, 228, 230, 231, 234, 240, 242, 243, 250, 252, 253, 260, 261, 264,
]  # fmt: skip


class BiddingStyle(Enum):
    """
    How much a bidder gambles on what the skat turns out to hold.

    The declared game is only worth what the ten cards *plus the skat* make it
    worth, and the soloist has to commit to a bid before seeing them. These are
    the sensible points on that gamble.
    """

    #: The value guaranteed however the skat falls - can never be überreizt.
    CONSERVATIVE = "conservative"
    #: The hand taken at face value, as if the skat changed nothing. Overbids
    #: exactly when the skat cuts a run of missing matadors short.
    NOMINAL = "nominal"
    #: The best the skat could possibly turn the hand into. Usually overbids.
    OPTIMISTIC = "optimistic"
    #: Ignore the hand entirely and say any rung on the ladder.
    RANDOM = "random"


def ceiling(
    style: BiddingStyle,
    game: Game,
    cards: set[Card],
    rng: Optional[random.Random] = None,
) -> int:
    """The highest value a bidder of this `style` will say for `game`."""
    if style is BiddingStyle.RANDOM:
        return (rng or random).choice(VALID_BIDS)
    low, high = game.spielwert_range(cards)
    match style:
        case BiddingStyle.CONSERVATIVE:
            return low
        case BiddingStyle.NOMINAL:
            return game.spielwert(cards)
        case BiddingStyle.OPTIMISTIC:
            return high
        case _:
            raise ValueError(f"unknown bidding style: {style}")


def next_rung(current_bid: int, limit: int) -> int:
    """
    The next value above `current_bid` a bidder capped at `limit` can say, or
    `0` to pass. `bisect` rather than `.index` so an off-ladder `current_bid`
    can't raise.
    """
    index = bisect.bisect_right(VALID_BIDS, current_bid)
    if index < len(VALID_BIDS) and VALID_BIDS[index] <= limit:
        return VALID_BIDS[index]
    return 0


# ----------------------------------------------------------- hand evaluation


def suit_length(cards: set[Card], suit: int) -> int:
    """Number of trumps a suit game in `suit` would give us (jacks included)."""
    return sum(1 for card in cards if card.is_jack or card.suit == SUITS[suit])


def trump_count(cards: set[Card], game: Game) -> int:
    trumps = set(game.trump_cards(game.suit))
    return sum(1 for card in cards if card in trumps)


def aces(cards: set[Card]) -> int:
    return sum(1 for card in cards if card.rank == "A")


def best_game(cards: set[Card]) -> Game:
    """
    The game these cards want to be played as.

    Suit choice goes by trump length — the multiplier alone would always favour
    clubs — with ties broken towards the richer suit. Grand wins out only on a
    hand carried by its jacks rather than by trump length: comparing the two
    Spielwerte would not do, since grand's base of 24 flatters every jack-heavy
    hand even when there is an obvious long suit to make trump of instead.

    Null is never chosen. Judging it needs the opposite reading of a hand (low
    cards, no length) and shares nothing with this estimate.
    """
    best_suit = max(range(4), key=lambda suit: (suit_length(cards, suit), suit))
    jacks = sum(1 for card in cards if card.is_jack)
    plain = suit_length(cards, best_suit) - jacks  # trumps that are not jacks
    return Grand() if jacks >= 3 and plain <= 3 else SuitGame(best_suit)


def playable(cards: set[Card]) -> bool:
    """Whether the hand is worth bidding on at all, style aside."""
    trumps = trump_count(cards, best_game(cards))
    return trumps >= 5 or (trumps >= 4 and aces(cards) >= 2)


def covering_game(cards: set[Card], minimum: int) -> Game:
    """
    The game to declare with `cards` (the ten cards *plus the skat*) that is
    worth at least `minimum` — what was bid.

    Declaring the hand's natural game is not automatically safe, even for a
    bidder that never overbid on its own hand: picking the skat up can make a
    different suit the longest, and a cheaper base multiplier can leave the
    game worth less than the bid. So prefer the natural game, fall back to the
    most playable game that does cover the bid, and if nothing covers it take
    the most valuable game there is — überreizt either way, but by less.
    """
    candidates: list[Game] = [SuitGame(suit) for suit in range(4)] + [Grand()]
    natural = best_game(cards)
    if natural.spielwert(cards) >= minimum:
        return natural
    by_playability = sorted(
        candidates,
        key=lambda game: (trump_count(cards, game), game.value),
        reverse=True,
    )
    for game in by_playability:
        if game.spielwert(cards) >= minimum:
            return game
    return max(candidates, key=lambda game: game.spielwert(cards))
