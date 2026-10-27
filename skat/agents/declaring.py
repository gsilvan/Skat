"""Interchangeable automatic strategies for pressing and declaring."""

from collections import Counter
from dataclasses import dataclass, replace

from skat.agents.bidding import best_game, covering_game
from skat.card import Card
from skat.games import Game


@dataclass(frozen=True)
class DeclarationChoice:
    pressed: tuple[Card, Card]
    game: Game


def validate_pressed(view, pressed):
    pressed = tuple(pressed)
    if (
        len(view.hand) != 12
        or len(pressed) != 2
        or len(set(pressed)) != 2
        or not set(pressed) <= set(view.hand)
    ):
        raise ValueError("press exactly two distinct cards from the twelve-card hand")
    return pressed


class HeuristicDeclarer:
    """The original conservative pickup, press and declaration baseline."""

    def choose_pickup(self, view):
        return True

    def choose_after_pickup(self, view, *, pressed=None):
        game = covering_game(set(view.hand), view.highest_bid)
        pair = (
            tuple(press_for_game(view.hand, game))
            if pressed is None
            else validate_pressed(view, pressed)
        )
        return DeclarationChoice(pair, game)

    def choose_hand(self, view):
        game = best_game(set(view.hand))
        game.declaration = replace(game.declaration, hand=True)
        return game


def press_for_game(cards, game):
    cards = list(cards)
    trumps = set(game.trump_cards(game.suit))
    lengths = Counter(card.suit for card in cards if card not in trumps)
    cards.sort(
        key=lambda card: (
            card in trumps,
            card.rank == "A",
            lengths[card.suit],
            -card.value,
            card.np_index,
        )
    )
    return cards[:2]
