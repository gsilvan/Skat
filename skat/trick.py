from typing import Union

from skat.card import Card
from skat.games import Game


class Trick:
    def __init__(self, game_mode: Game) -> None:
        self.card_turn: list[tuple[int, Card]] = list()
        self.game_mode: Game = game_mode

    def __len__(self) -> int:
        return len(self.card_turn)

    def append(self, player_id: int, card: Card) -> None:
        """Adds a (player, card)-tuple to current trick"""
        if len(self.card_turn) < 3:
            self.card_turn.append((player_id, card))
        else:
            raise Exception("can't add more than 3 cards for a single trick")
            # TODO: Use a more specific Exception.

    @property
    def color(self):
        """Returns the color of this trick."""
        if len(self.card_turn) > 0:
            return self.card_turn[0][1].suit
        else:
            return None

    @property
    def is_full(self) -> bool:
        """
        Returns true if the trick is full, meaning all players placed their
        card. Returns false if cards missing."""
        return len(self.card_turn) == 3

    @property
    def value(self) -> int:
        """Returns the {current, final} value of a trick"""
        _sum = 0
        for _, card in self.card_turn:
            _sum += card.value
        return _sum

    @property
    def winner(self) -> Union[int, None]:
        """Returns the trick winner's player_id"""
        if self.is_full:
            return self.game_mode.trick_winner(self)
        else:
            return None
