from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from skat.hand import Hand

if TYPE_CHECKING:
    from skat.agents import Agent
    from skat.card import Card
    from skat.games import Game
    from skat.trick import Trick


class Player:
    def __init__(self, agent: Agent, seat_id: int) -> None:
        """Initialize a Skat player with a given agent strategy."""
        self.strategy: Agent = agent
        self.strategy.set_state(state=self)  # we give our agent a pointer
        self.seat_id: int = seat_id
        self.hand: Hand = Hand()
        self.trick_stack: Hand = Hand()
        self.public_state: Optional[Game] = None

    def __str__(self) -> str:
        return f"{self.seat_id} hand={self.hand} " f"score={self.trick_stack_value}"

    @property
    def trick_stack_value(self) -> int:
        """Returns the players current trick value."""
        return self.trick_stack.value

    def valid_moves(self, trick: Trick):
        """Returns a set of valid moves given a trick and a hand"""
        card_set = set(self.hand) & trick.forced_cards
        if len(card_set) == 0:
            return set(self.hand)
        else:
            return card_set

    def receive_cards(self, cards: list[Card]):
        """Append the received cards from the dealer to the own hand."""
        for card in cards:
            self.hand.append(card)

    def bid(self, current_bid: int, offer=False) -> int:
        """
        Make a bid given a current bid. This function delegates the bidding to
        the specified agent. The agent returns a new bid or folds.
        """
        return self.strategy.bid(current_bid, offer)

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

    def play_card(self, current_trick: Trick) -> Card:
        """Select the next card to play. Delegated to the agent strategy."""
        valid_moves = self.valid_moves(trick=current_trick)
        return self.strategy.choose_card(valid_moves)

    def take_trick(self, trick) -> None:
        """Append won trick to the own trick stack."""
        for _, card in trick.buffer:
            self.trick_stack.append(card)

    def set_state(self, state) -> None:
        self.public_state = state
