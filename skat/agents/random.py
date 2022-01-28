import random
from typing import Union

from skat.agents import Agent
from skat.card import Card
from skat.games import Game
from skat.games.suit import SuitGame
from skat.player import Player


class RandomAgent(Agent):
    """
    Random-Agent: An agent that identifies valid moves and chooses on of those
    by an equal distributed random policy.
    """

    def __init__(self) -> None:
        """Initializes the random agent with a fixed seed."""
        self.state: Union[Player, None] = None
        self.max_bid = random.choice(self.VALID_BIDS)
        print(self.max_bid)

    def bid(self, current_bid) -> int:
        """Random Agent selects a bid from a list containing valid bids"""
        bid_index = self.VALID_BIDS.index(current_bid)
        max_bid_index = self.VALID_BIDS.index(self.max_bid)
        if bid_index < max_bid_index:
            return self.VALID_BIDS[bid_index + 1]
        return 0

    def pickup_skat(self, state) -> bool:
        """Random Agent selects skats equal distributed"""
        return bool(random.getrandbits(1))

    def press_skat(self) -> list[Card]:
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        skat = list()
        for i in range(2):
            skat.append(self.state.hand.pop(
                random.randint(0, 11 - i)))
        return skat

    def declare_game(self, state) -> Game:
        """Random Agent declares games equal distributed"""
        available_games = (
            SuitGame(random.randint(0, 3)),
        )
        return random.choice(available_games)

    def choose_card(self) -> Card:
        """Random Agent chooses a random valid move"""
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        return self.state.hand.pop(
            random.randint(0, len(self.state.hand) - 1))

    def set_state(self, state: Player) -> None:
        self.state = state
