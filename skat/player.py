from skat.agents import Agent
from skat.card import Card
from skat.games import Game


class Player:
    player_count = 0

    def __init__(self, agent: Agent) -> None:
        self.strategy: Agent = agent
        self.seat_id: int = Player.player_count
        self.hand: list[Card] = list()
        self.tricks: list[Card] = list()
        Player.player_count += 1

    def __del__(self) -> None:
        Player.player_count -= 1

    @property
    def trick_value(self) -> int:
        """Returns players current trick value"""
        _sum = 0
        for card in self.tricks:
            _sum += card.value
        return _sum

    def bid(self) -> int:
        return self.strategy.bid(None)

    def pickup_skat(self) -> bool:
        return self.strategy.pickup_skat(None)

    def declare_game(self) -> Game:
        return self.strategy.declare_game(None)

    def play_card(self):
        self.strategy.choose_card(None)
