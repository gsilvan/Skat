import random

from skat.card import Card
from skat.games import Game
from skat.games.suit import SuitGame

from . import Agent


class RandomAgent(Agent):
    """
    Random-Agent: An agent that identifies valid moves and chooses on of those
    by an equal distributed random policy.
    """
    def __init__(self):
        """Initializes the random agent with a fixed seed."""
        random.seed(0)

    def bid(self, state) -> int:
        """Random Agent selects a bid from a list containing valid bids"""
        valid_bids = (
            18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 48
        )
        return random.choice(valid_bids)

    def pickup_skat(self, state) -> bool:
        """Random Agent selects skats equal distributed"""
        return bool(random.getrandbits(1))

    def declare_game(self, state) -> Game:
        """Random Agent declares games equal distributed"""
        available_games = (
            SuitGame(random.randint(0, 3)),
        )
        return random.choice(available_games)

    def choose_card(self, state) -> Card:
        """Random Agent chooses a random valid move"""
        pass
