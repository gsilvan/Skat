from skat.card import RANKS, SUITS, Card
from skat.games import Game
from skat.trick import Trick

MULTIPLIERS = (9, 10, 11, 12)


class SuitGame(Game):
    """Implements rules for a Suit-Game"""
    def __init__(self, suit):
        self.suit = suit

    def trumps(self) -> list[Card]:
        """Returns a list of trumps for the selected game"""
        trumps = list()
        for j in range(len(SUITS)):
            trumps.append(Card(j, 3))  # {♦J, ♥J, ♠J, ♣J}
        for i, rank in enumerate(RANKS):
            if i == 3:
                continue  # skip J, because he's already in the list
            trumps.append(Card(self.suit, i))
        return trumps

    @property
    def value(self) -> int:
        """Returns the base multiplier for the selected game"""
        return MULTIPLIERS[self.suit]

    @staticmethod
    def evaluate_trick(trick: Trick) -> tuple[int, int]:
        """Trick evaluation for Suit-Game"""
        # TODO: Implementation
        return 0, 0
