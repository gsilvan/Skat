"""Conservative automatic declaration, independent of card-play strategy."""

from collections import Counter
from dataclasses import replace

from skat.agents.bidding import best_game, covering_game


class HeuristicDeclarer:
    def intended_game(self):
        return best_game(set(self.state.hand))

    def pickup_skat(self, state=None):
        return True

    def press_skat(self):
        game = covering_game(set(self.state.hand), self.state.highest_bid)
        return press_for_game(self.state.hand, game)

    def declare_game(self, state=None):
        if self.state.hand_game:
            game = self.intended_game()
            game.declaration = replace(game.declaration, hand=True)
            return game
        return covering_game(
            set(self.state.hand) | set(self.state.skat), self.state.highest_bid
        )


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
