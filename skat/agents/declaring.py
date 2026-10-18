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


class HeuristicDeclarer:
    """The original conservative pickup, press and declaration baseline."""

    def choose_pickup(self, view):
        return True

    def choose_after_pickup(self, view):
        game = covering_game(set(view.hand), view.highest_bid)
        return DeclarationChoice(tuple(press_for_game(view.hand, game)), game)

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
