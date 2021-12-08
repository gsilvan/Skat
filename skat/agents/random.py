import random

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
    VALID_BIDS = [0, 18, 20, 22, 24, 27, 30, 33, 36, 40, 44, 45, 48, 50, 54,
                  55,
                  60, 63, 66, 70, 72, 77, 80, 81, 84, 88, 90, 96, 99, 100, 108,
                  110, 117, 120, 121, 126, 130, 132, 135, 140, 143, 144, 150,
                  153, 154, 156, 160, 162, 165, 168, 170, 171, 176, 180, 187,
                  189, 190, 192, 198, 200, 204, 207, 209, 210, 216, 220, 225,
                  228, 230, 231, 234, 240, 242, 243, 250, 252, 253, 260, 261,
                  264]

    def __init__(self):
        """Initializes the random agent with a fixed seed."""
        self.state: [Player, None] = None
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
        return self.state.hand.pop(
            random.randint(0, len(self.state.hand) - 1))

    def set_state(self, state: Player) -> None:
        self.state: Player = state
