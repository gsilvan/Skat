from enum import auto, Enum

from skat.agents.random import RandomAgent
from skat.player import Player
from skat.deck import Deck


class GamePhase(Enum):
    WAITING = auto()
    DEALING = auto()
    BIDDING = auto()
    DECLARING = auto()
    PLAYING = auto()
    COUNTING = auto()


class Round:
    def __init__(self):
        self._phase: GamePhase = GamePhase.WAITING
        self._player: list[Player] = list()
        self._back_hand: int = 0  # aka the "dealer"
        self._highest_bid = 0
        self._highest_bid_seat_id = -42
        self._deck = list()
        self.init_players()

    @property
    def phase(self):
        return self._phase

    def init_players(self):
        for _ in range(3):
            self._player.append(Player(RandomAgent()))

    def start(self):
        # Wait for players
        while self._phase == GamePhase.WAITING:
            if len(self._player) == 3:
                self._phase = GamePhase.DEALING
        # card dealing
        self._deck = Deck()
        self._deck.initialize_cards()
        self._deck.shuffle()
        for player in self._player:
            player.hand = self._deck.deal_cards()
        self._phase = GamePhase.BIDDING
        # game bidding
        front_hand = (self._back_hand + 1) % 3  # Vorhand
        middle_hand = (self._back_hand + 2) % 3  # Mittelhand
        back_hand = self._back_hand  # Hinterhand
        while True:
            middle_hand_bid = self._player[middle_hand].bid(self._highest_bid)
            if middle_hand_bid > self._highest_bid:
                self._highest_bid = middle_hand_bid
                self._highest_bid_seat_id = middle_hand
            else:
                break
            front_hand_bid = self._player[front_hand].bid(self._highest_bid)
            if front_hand_bid > self._highest_bid:
                self._highest_bid = front_hand_bid
                self._highest_bid_seat_id = front_hand
            else:
                break
        stage_one_winner = self._highest_bid_seat_id
        while True:
            back_hand_bid = self._player[back_hand].bid(self._highest_bid)
            if back_hand_bid > self._highest_bid:
                self._highest_bid = back_hand_bid
                self._highest_bid_seat_id = back_hand
            else:
                break
            stage_one_winner_bid = self._player[stage_one_winner].bid(
                self._highest_bid)
            if stage_one_winner_bid > self._highest_bid:
                self._highest_bid = stage_one_winner_bid
                self._highest_bid_seat_id = stage_one_winner
            else:
                break
