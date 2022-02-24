from typing import Optional

from skat.card import RANKS, SUITS, Card
from skat.game import Game
from skat.trick import Trick


class NullTrick(Trick):
    @property
    def forced_cards(self) -> set[Card]:
        if len(self.card_turn) == 0:
            return set()
        else:
            return set(Null.suit_cards(SUITS.index(self.card_turn[0][1].suit)))

    @property
    def is_trump(self) -> bool:
        """
        There are no trumps in null game, therefore a trick is always a non
        trump-trick
        """
        return False

    @property
    def winner(self) -> Optional[int]:
        if not self.is_full:
            return None
        winner: tuple[int, Card] = self.card_turn[0]
        for player_card in self.card_turn[1:]:
            if player_card[1].suit == self.card_turn[0][1].suit:
                if self.better_than(player_card[1], winner[1]):
                    winner = player_card
        return winner[0]

    @staticmethod
    def better_than(a: Card, b: Card):
        """Test if card `a` is better than card `b`"""
        if a.suit != b.suit:
            raise Exception("Can't compare non equal suits")
        order = ("7", "8", "9", "X", "J", "Q", "K", "A")
        return order.index(a.rank) > order.index(b.rank)


class Null(Game):
    def __init__(self) -> None:
        self.trick = NullTrick()

    def new_trick(self) -> None:
        self.trick = NullTrick()

    @staticmethod
    def trump_cards() -> tuple[Card, ...]:
        """There are no trumps, return always empty tuple"""
        return tuple()

    @staticmethod
    def suit_cards(suit) -> tuple[Card, ...]:
        suits = list()
        for i, _ in enumerate(RANKS):
            suits.append(Card(suit, i))
        return tuple(suits)

    @property
    def value(self) -> int:
        return 23
