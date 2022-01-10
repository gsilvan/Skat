from enum import Enum, auto

from skat.agents.random import RandomAgent
from skat.card import Card
from skat.deck import Deck
from skat.games import Game
from skat.player import Player
from skat.trick import Trick


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
        self._highest_bid: int = 0
        self._highest_bid_seat_id: int = -42
        self._deck: list[Card] = list()
        self._won_last_trick: int = (self._back_hand + 1) % 3
        self._hand_game: bool = False
        self._skat: list[Card] = list()
        self._game: [Game, None] = None
        self._trick: [Trick, None] = None
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
        # take skat or not
        self._hand_game = not self._player[
            self._highest_bid_seat_id].pickup_skat()
        if not self._hand_game:
            _cards_in_skat = self._deck.deal_cards(2)
            print(
                f"p={self._highest_bid_seat_id} picks the skat: "
                f"{_cards_in_skat}")
            self._player[self._highest_bid_seat_id].receive_cards(
                _cards_in_skat
            )
            self._skat = self._player[self._highest_bid_seat_id].press_skat()
            print(f"p={self._highest_bid_seat_id} puts {self._skat} in skat")
        else:
            print(f"p={self._highest_bid_seat_id} discards the skat")
            self._skat = self._deck.deal_cards(2)
            print(f"skat={self._skat}")
        for p in self._player:
            print(p)
        print(f"skat={self._skat}")
        # game declaration
        self._game = self._player[self._highest_bid_seat_id].declare_game()
        # add skat to tricks
        for card in self._skat:
            self._player[self._highest_bid_seat_id].tricks.append(card)
        # card_outplay
        while any(len(p.hand) for p in self._player):
            self._trick = Trick(self._game)
            self._trick.append(self._won_last_trick,
                               self._player[self._won_last_trick].play_card())
            self._trick.append(self._won_last_trick,
                               self._player[
                                   (self._won_last_trick + 1) % 3].play_card())
            self._trick.append(self._won_last_trick,
                               self._player[
                                   (self._won_last_trick + 2) % 3].play_card())
            self._won_last_trick, _ = self._trick.winner()
            self._player[self._won_last_trick].take_trick(self._trick)
        # counting
        for p in self._player:
            print(f"p={p.seat_id} h={p.hand} points={p.trick_value}")
