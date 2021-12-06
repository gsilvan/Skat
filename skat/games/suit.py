from skat.card import RANKS, SUITS, Card
from skat.games import Game
from skat.trick import Trick

MULTIPLIERS = (9, 10, 11, 12)


class SuitGame(Game):
    """Implements rules for a Suit-Game"""
    def __init__(self, suit):
        self.suit = suit

    def trump_set(self) -> tuple[Card, ...]:
        # TODO: option for ascending, descending
        """Returns a ascending ordered tuple of trumps for the selected game"""
        trumps = list()
        for i, rank in enumerate(RANKS):
            if i == 3:
                continue  # skip J, we add him later.
            trumps.append(Card(self.suit, i))
        for j in range(len(SUITS)):
            trumps.append(Card(j, 3))  # add Js {♦J, ♥J, ♠J, ♣J}
        return tuple(trumps)

    @staticmethod
    def suit_set(suit) -> tuple[Card, ...]:
        # TODO: option for ascending, descending
        suits = list()
        for i, _ in enumerate(RANKS):
            if i == 3:
                continue  # skip J
            suits.append(Card(suit, i))
        return tuple(suits)

    def is_trump_card(self, card: Card) -> bool:
        """Check if given card is a trump card"""
        card_set: set = {card}  # put the card in a set and compare it
        return card_set < set(self.trump_set())

    @staticmethod
    def is_jack_card(card: Card) -> bool:
        """Return True if given card is a J"""
        return card.rank == 3

    def is_trump_in_trick(self, trick: Trick) -> bool:
        return any(x in self.trump_set() for _, x in trick.card_outplays)

    @property
    def value(self) -> int:
        """Returns the base multiplier for the selected game"""
        return MULTIPLIERS[self.suit]

    @staticmethod
    def evaluate_trick(trick: Trick) -> tuple[int, int]:
        """Trick evaluation for Suit-Game"""
        # TODO: Implementation
        return 0, 0

    def trick_winner(self, trick) -> int:
        winner: tuple = trick.card_outplays[0]
        if self.is_trump_in_trick(trick):
            # then only here can be the winner
            for player_card in trick.card_outplays[1:]:
                if self.is_trump_card(player_card[1]):
                    if self.is_jack_card(player_card[1]):
                        if self.is_jack_card(winner[1]):
                            if player_card[1].suit > winner[1].suit:
                                winner = player_card
                        else:
                            winner = player_card
                    else:
                        if player_card[1] > winner[1]:
                            winner = player_card
            return winner[0]
        else:
            for player_card in trick.card_outplays[1:]:
                if player_card[1].suit == trick.card_outplays[0][1].suit:
                    if player_card[1] > winner[1]:
                        winner = player_card
            return winner[0]
