from typing import Optional

from skat.card import RANKS, SUITS, Card
from skat.game import Game
from skat.trick import Trick


class GrandTrick(Trick):
    @property
    def forced_cards(self) -> set[Card]:
        if len(self.card_turn) == 0:
            return set()
        else:
            if self.is_trump:
                return set(Grand.trump_cards())
            else:
                return set(Grand.suit_cards(SUITS.index(self.card_turn[0][1].suit)))

    @property
    def is_trump(self) -> bool:
        if len(self.card_turn) <= 0:
            return False
        return self.card_turn[0][1].is_jack

    @property
    def winner(self) -> Optional[int]:
        if not self.is_full:
            return None
        winner: tuple[int, Card] = self.card_turn[0]
        if self._is_trump_in_trick():
            for player_card in self.card_turn[1:]:
                if player_card[1].is_jack:
                    if winner[1].is_jack:
                        if winner[1] < player_card[1]:
                            winner = player_card
                    else:
                        winner = player_card
            return winner[0]
        else:
            for player_card in self.card_turn[1:]:
                if player_card[1].suit == self.card_turn[0][1].suit:
                    if player_card[1] > winner[1]:
                        winner = player_card
            return winner[0]

    def _is_trump_in_trick(self) -> bool:
        return any(x.is_jack for _, x in self.card_turn)


class Grand(Game):
    def __init__(self) -> None:
        self.trick = GrandTrick()

    def new_trick(self) -> None:
        self.trick = GrandTrick()

    @staticmethod
    def trump_cards(suit=None) -> tuple[Card, ...]:
        trumps = list()
        for i, _ in enumerate(SUITS):
            trumps.append(Card(i, 3))
        return tuple(trumps)

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        suits = list()
        for i, _ in enumerate(RANKS):
            if i == 3:
                continue  # skip J
            suits.append(Card(suit, i))
        return tuple(suits)

    @property
    def value(self) -> int:
        return 24
