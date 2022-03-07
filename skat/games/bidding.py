from skat.player import Player


class BiddingGame:
    def __init__(self, players: list[Player]):
        if len(players) != 3:
            raise Exception("provide three players")
        if not any(isinstance(p, Player) for p in players):
            raise Exception("only player objects allowed")
        for p in players:
            setattr(p, "has_fold", False)
        self.front_hand = players[0]
        self.middle_hand = players[1]
        self.back_hand = players[2]
        self.bid_history: list = list()
        self.bid = 0
        for p in players:
            print(getattr(p, "has_fold"))

    def start(self):
        bid = self.middle_hand.bid(current_bid=self.bid, offer=False)
        bid_2 = self.front_hand.bid(current_bid=self.bid, offer=True)


if __name__ == "__main__":
    from skat.agents.random import RandomAgent
    from skat.player import Player

    agents = [
        Player(RandomAgent(), 0),
        Player(RandomAgent(), 0),
        Player(RandomAgent(), 0),
    ]
    bidding_game = BiddingGame(agents)
