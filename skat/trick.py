from skat.card import Card
from skat.games import Game


class Trick:
    def __init__(self, game_mode: Game) -> None:
        self.card_outplays: list[tuple[int, Card]] = list()
        self.game_mode: Game = game_mode
        self.trick_color = ''  # Holds the suit of card_outplays[0]

    def __len__(self) -> int:
        return len(self.card_outplays)

    def append(self, player_id: int, card: Card) -> None:
        """Adds a (player, card)-tuple to current trick"""
        if len(self.card_outplays) < 3:
            if len(self.card_outplays) == 0:
                self.trick_color = card.suit
            self.card_outplays.append((player_id, card))
        else:
            raise Exception("can't add more than 3 cards for a single trick")
            # TODO: Use a more specific Exception.

    @property
    def value(self) -> int:
        """Returns the {current, final} value of a trick"""
        _sum = 0
        for _, card in self.card_outplays:
            _sum += card.value
        return _sum

    @property
    def is_full(self) -> bool:
        """
        Returns true if the trick is full, meaning all players placed their
        card. Returns false if cards missing."""
        return len(self.card_outplays) == 3

    def winner(self) -> tuple[int, int]:
        """
        Evaluates the winner of the current trick
        :return: trick winner's player_id, score
        """
        if self.is_full:
            return self.game_mode.evaluate_trick(self)
        else:
            raise Exception("can't evaluate the trick with only "
                            f"{len(self.card_outplays)} cards in trick.")
            # TODO: Use a more specific Exception
