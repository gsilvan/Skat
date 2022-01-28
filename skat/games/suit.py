from typing import Union

from skat.card import RANKS, SUITS, Card
from skat.games import Game
from skat.trick import Trick

MULTIPLIERS = (9, 10, 11, 12)


class SuitGameTrick(Trick):
    def __init__(self, suit):
        super().__init__()
        self.trump_suit = suit

    def _is_trump_card(self, card: Card) -> bool:
        """Check if given card is a trump card"""
        card_set: set = {card}  # put the card in a set and compare it
        return card_set < set(SuitGame.trump_cards(self.trump_suit))

    def _is_trump_in_trick(self) -> bool:
        return any(x in SuitGame.trump_cards(self.trump_suit) for _, x in
                   self.card_turn)

    @property
    def forced_cards(self) -> set[Card]:
        if len(self.card_turn) == 0:
            return set()
        else:
            if self.is_trump:
                return set(SuitGame.trump_cards(self.trump_suit))
            else:
                return set(SuitGame.suit_cards(
                    SUITS.index(self.card_turn[0][1].suit)))

    @property
    def is_trump(self) -> bool:
        if len(self.card_turn) <= 0:
            raise Exception("No card in trick")
        return self._is_trump_card(self.card_turn[0][1])

    @property
    def winner(self) -> Union[int, None]:
        """Returns the trick winner's player_id"""
        if not self.is_full:
            return None
        winner: tuple = self.card_turn[0]
        if self._is_trump_in_trick():
            # then only here can be the winner
            for player_card in self.card_turn[1:]:
                if self._is_trump_card(player_card[1]):
                    if player_card[1].is_jack:
                        if winner[1].is_jack:
                            if winner[1] < player_card[1]:
                                winner = player_card
                        else:
                            winner = player_card
                    else:
                        if player_card[1] > winner[1]:
                            winner = player_card
            return winner[0]
        else:
            for player_card in self.card_turn[1:]:
                if player_card[1].suit == self.card_turn[0][1].suit:
                    if player_card[1] > winner[1]:
                        winner = player_card
            return winner[0]


class SuitGame(Game):
    """Implements rules for a Suit-Game"""

    def __init__(self, suit):
        self.suit = suit
        self.trick = SuitGameTrick(self.suit)

    def new_trick(self):
        self.trick = SuitGameTrick(self.suit)

    @staticmethod
    def trump_cards(suit) -> tuple[Card, ...]:
        # TODO: option for ascending, descending
        """
        Returns an ascending ordered tuple of trumps for the selected game
        """
        trumps = list()
        for i, rank in enumerate(RANKS):
            if i == 3:
                continue  # skip J, we add him later.
            trumps.append(Card(suit, i))
        for j in range(len(SUITS)):
            trumps.append(Card(j, 3))  # add Js {♦J, ♥J, ♠J, ♣J}
        return tuple(trumps)

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        # TODO: option for ascending, descending
        suits = list()
        for i, _ in enumerate(RANKS):
            if i == 3:
                continue  # skip J
            suits.append(Card(suit, i))
        return tuple(suits)

    @property
    def value(self) -> int:
        """Returns the base multiplier for the selected game"""
        return MULTIPLIERS[self.suit]
