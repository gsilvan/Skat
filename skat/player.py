from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skat.agents import Agent
    from skat.card import Card
    from skat.games import Game


class Player:
    player_count: int = 0

    def __init__(self, agent: Agent) -> None:
        """Initialize a Skat player with a given agent strategy."""
        self.strategy: Agent = agent
        self.strategy.set_state(state=self)  # we give our agent a pointer
        self.seat_id: int = Player.player_count
        self.hand: list[Card] = list()
        self.tricks: list[Card] = list()
        self.public_state = None
        Player.player_count += 1

    def __del__(self) -> None:
        Player.player_count -= 1

    def __str__(self) -> str:
        return f"{self.seat_id} hand={self.hand} " \
               f"score={self.trick_stack_value}"

    @property
    def trick_stack_value(self) -> int:
        """Returns the players current trick value."""
        _sum = 0
        for card in self.tricks:
            _sum += card.value
        return _sum

    def receive_cards(self, cards: list[Card]):
        """Append the received cards from the dealer to the own hand."""
        for card in cards:
            self.hand.append(card)

    def bid(self, current_bid: int) -> int:
        """
        Make a bid given a current bid. This function delegates the bidding to
        the specified agent. The agent returns a new bid or folds.
        """
        return self.strategy.bid(current_bid)

    def pickup_skat(self) -> bool:
        """
        Pick up the Skat or play a hand-game. This function delegates the skat
        taking decision to the specified agent. The agent returns a bool.
        """
        return self.strategy.pickup_skat(None)

    def press_skat(self) -> list[Card]:
        """Returns 2 Cards to put them in the Skat. Delegated to the agent."""
        return self.strategy.press_skat()

    def declare_game(self) -> Game:
        """Game declaration. Delegated to the agent strategy."""
        return self.strategy.declare_game(None)

    def play_card(self) -> Card:
        """Select the next card to play. Delegated to the agent strategy."""
        return self.strategy.choose_card()

    def take_trick(self, trick) -> None:
        """Append won trick to the own trick stack."""
        for _, card in trick.card_turn:
            self.tricks.append(card)

    def set_state(self, state) -> None:
        self.public_state = state
