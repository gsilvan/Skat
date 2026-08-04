from __future__ import annotations

import random
from typing import TYPE_CHECKING

from skat.agents import Agent
from skat.agents.bidding import BiddingStyle
from skat.card import Card
from skat.games.suit import SuitGame

if TYPE_CHECKING:
    from skat.games import Game


class RandomAgent(Agent):
    """
    Random-Agent: An agent that identifies valid moves and chooses on of those
    by an equal distributed random policy.

    Bidding is the one decision it does not make at random by default: it
    inherits `BiddingStyle.CONSERVATIVE`, because a tournament reuses one agent
    object per seat and a ceiling drawn from the whole ladder (up to 264) wins
    every auction and then loses the round to überreizt. Pass
    `BiddingStyle.RANDOM` for the uniform-random bidder.
    """

    def __init__(self, bidding: BiddingStyle = BiddingStyle.CONSERVATIVE) -> None:
        super().__init__(bidding=bidding)
        self.game: Game = SuitGame(random.randint(0, 3))

    def new_round_event(self) -> None:
        super().new_round_event()
        # Settle the declaration up front so the hand is valued for the game we
        # will actually declare - otherwise a conservative ceiling means nothing.
        self.game = SuitGame(random.randint(0, 3))

    def intended_game(self) -> Game:
        return self.game

    def declare_game(self, state) -> Game:
        """Random Agent declares games equal distributed"""
        return self.game

    def pickup_skat(self, state) -> bool:
        """Random Agent selects skats equal distributed"""
        return bool(random.getrandbits(1))

    def press_skat(self) -> list[Card]:
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        skat = list()
        for i in range(2):
            skat.append(self.state.hand.pop(random.randint(0, 11 - i)))
        return skat

    def choose_card(self, valid_moves: set[Card]) -> Card:
        """Random Agent chooses a random valid move"""
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        choice = random.choice(list(valid_moves))
        self.state.hand.remove(choice)
        return choice
