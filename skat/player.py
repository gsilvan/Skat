from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skat.agents import Agent
    from skat.card import Card
    from skat.games import Game


class Player:
    player_count = 0

    def __init__(self, agent: Agent) -> None:
        self.strategy: Agent = agent
        self.strategy.set_state(state=self)  # we give our agent a pointer
        self.seat_id: int = Player.player_count
        self.hand: list[Card] = list()
        self.tricks: list[Card] = list()
        Player.player_count += 1

    def __del__(self) -> None:
        Player.player_count -= 1

    def __str__(self) -> str:
        return f"{self.seat_id} hand={self.hand} score={self.trick_value}"

    @property
    def trick_value(self) -> int:
        """Returns players current trick value"""
        _sum = 0
        for card in self.tricks:
            _sum += card.value
        return _sum

    def receive_cards(self, cards: list[Card]):
        for card in cards:
            self.hand.append(card)

    def bid(self, current_bid: int) -> int:
        return self.strategy.bid(current_bid)

    def pickup_skat(self) -> bool:
        return self.strategy.pickup_skat(None)

    def press_skat(self) -> list[Card]:
        return self.strategy.press_skat()

    def declare_game(self) -> Game:
        return self.strategy.declare_game(None)

    def play_card(self) -> Card:
        return self.strategy.choose_card()

    def take_trick(self, trick) -> None:
        for _, card in trick.card_outplays:
            self.tricks.append(card)
