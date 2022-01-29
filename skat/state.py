import copy
from enum import Enum, auto
from typing import Union

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
    def __init__(self, dealer: int = 0) -> None:
        self._phase: GamePhase = GamePhase.WAITING
        self._player: list[Player] = list()
        self._dealer: int = dealer
        self._highest_bid: int = 0
        self._highest_bid_seat_id: int = -42
        self._deck: Deck = Deck()
        self._hand_game: bool = False
        self._skat: list[Card] = list()
        self._game: Union[Game, None] = None
        self._trick_history: list[Trick] = list()
        self.init_players()

    @property
    def front_hand(self) -> int:
        if len(self._trick_history) > 0:
            if self._trick_history[-1].winner is None:
                raise Exception("There is no winner yet.")
            return self._trick_history[-1].winner
        else:
            return (self._dealer + 1) % 3

    @property
    def middle_hand(self) -> int:
        return (self.front_hand + 1) % 3

    @property
    def back_hand(self) -> int:
        return (self.front_hand + 2) % 3

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def trick_history(self) -> list[Trick]:
        return self._trick_history

    def init_players(self) -> None:
        for _ in range(3):
            self._player.append(Player(RandomAgent()))
        for player in self._player:
            player.set_state(self)

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
        while True:
            middle_hand_bid = self._player[self.middle_hand].bid(self._highest_bid)
            if middle_hand_bid > self._highest_bid:
                self._highest_bid = middle_hand_bid
                self._highest_bid_seat_id = self.middle_hand
            else:
                break
            front_hand_bid = self._player[self.front_hand].bid(self._highest_bid)
            if front_hand_bid > self._highest_bid:
                self._highest_bid = front_hand_bid
                self._highest_bid_seat_id = self.front_hand
            else:
                break
        stage_one_winner = self._highest_bid_seat_id
        while True:
            back_hand_bid = self._player[self.back_hand].bid(self._highest_bid)
            if back_hand_bid > self._highest_bid:
                self._highest_bid = back_hand_bid
                self._highest_bid_seat_id = self.back_hand
            else:
                break
            stage_one_winner_bid = self._player[stage_one_winner].bid(self._highest_bid)
            if stage_one_winner_bid > self._highest_bid:
                self._highest_bid = stage_one_winner_bid
                self._highest_bid_seat_id = stage_one_winner
            else:
                break
        # take skat or not
        self._hand_game = not self._player[self._highest_bid_seat_id].pickup_skat()
        if not self._hand_game:
            _cards_in_skat = self._deck.deal_cards(2)
            print(f"p={self._highest_bid_seat_id} picks the skat: " f"{_cards_in_skat}")
            self._player[self._highest_bid_seat_id].receive_cards(_cards_in_skat)
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
            self._player[self._highest_bid_seat_id].trick_stack.append(card)
        # card_outplay
        while any(len(p.hand) for p in self._player):
            self._game.new_trick()
            self._game.trick.append(
                self.front_hand,
                self._player[self.front_hand].play_card(self._game.trick),
            )
            self._game.trick.append(
                self.middle_hand,
                self._player[self.middle_hand].play_card(self._game.trick),
            )
            self._game.trick.append(
                self.back_hand, self._player[self.back_hand].play_card(self._game.trick)
            )
            self._trick_history.append(copy.deepcopy(self._game.trick))
            self._player[self._game.trick.winner].take_trick(self._game.trick)
        # counting
        for p in self._player:
            print(f"p={p.seat_id} h={p.hand} points={p.trick_stack_value}")
