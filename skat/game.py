import copy
from enum import Enum, auto
from typing import Optional

from skat.agents.command_line import CommandLineAgent
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


class Tournament:
    def __init__(
        self,
        rounds=32,
        agents=(RandomAgent(), RandomAgent(), RandomAgent()),
    ) -> None:
        self.rounds = rounds
        self.agents = agents
        self.scores = [0, 0, 0]
        self.dealer = 0

    def start(self):
        for _ in range(self.rounds):
            r = Round(
                dealer=self.dealer,
                skip_bidding=True,
                solo_player_id=self.dealer,
                start=False,
                verbose=False,
                agents=self.agents,
            )
            soloist, points = r.start()
            self.scores[soloist] += points
            self.scores[(soloist + 1) % 3] += -points / 2
            self.scores[(soloist + 2) % 3] += -points / 2
            self.dealer = (self.dealer + 1) % 3


class Round:
    def __init__(
        self,
        dealer: int = 0,
        skip_bidding: bool = False,
        solo_player_id: int = -42,
        declare_game: Game = None,
        pickup_skat: bool = False,
        agents=None,
        deck=Deck(),
        start=True,
        verbose=False,
        seed=None,
    ) -> None:
        if agents is None:
            agents = list()
        self._player: list[Player] = list()
        self._phase: GamePhase = GamePhase.WAITING
        self._dealer: int = dealer
        self._highest_bid: int = 0
        self._highest_bid_seat_id: int = solo_player_id
        self._deck: Deck = deck
        self._verbose = verbose
        self._hand_game: bool = pickup_skat
        self._skat: list[Card] = list()
        self._game: Optional[Game] = declare_game
        self._skip_bidding: bool = skip_bidding
        self._trick_history: list[Trick] = list()
        self._seed = seed
        if len(agents) > 0 and len(agents) != 3:
            raise Exception("specify either 3 players or None")
        elif len(agents) == 0:
            # init default agents
            self.init_players()
        elif len(agents) == 3:
            self.init_players(agents)
        if start:
            # start the game with selected features immediately
            self.start()

    @property
    def front_hand(self) -> int:
        """Return player_id in front-hand-position"""
        if len(self._trick_history) > 0:
            if self._trick_history[-1].winner is None:
                raise Exception("There is no winner yet.")
            return self._trick_history[-1].winner
        else:
            return (self._dealer + 1) % 3

    @property
    def middle_hand(self) -> int:
        """Return player_id in middle-hand-position"""
        return (self.front_hand + 1) % 3

    @property
    def back_hand(self) -> int:
        """Return player_id in back-hand-position"""
        return (self.front_hand + 2) % 3

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def trick_history(self) -> list[Trick]:
        return self._trick_history

    @property
    def points_soloist(self) -> int:
        soloist = self._highest_bid_seat_id
        if soloist == -42:
            return 0
        return self._player[soloist].trick_stack_value

    @property
    def points_defenders(self):
        soloist = self._highest_bid_seat_id
        if soloist == -42:
            return 0
        return (
            self._player[(soloist + 1) % 3].trick_stack_value
            + self._player[(soloist + 2) % 3].trick_stack_value
        )

    def init_players(self, agents=None) -> None:
        if not agents:
            for i in range(3):
                self._player.append(Player(RandomAgent(), i))
        else:
            for idx, agent in enumerate(agents):
                self._player.append(Player(agent, idx))
        for player in self._player:
            player.set_state(self)

    @property
    def next_player(self) -> Optional[int]:
        """Next player to play a card"""
        if len(self._game.trick) == 0:
            return self.front_hand
        elif len(self._game.trick) == 1:
            return self.middle_hand
        elif len(self._game.trick) == 2:
            return self.back_hand
        elif len(self._game.trick) == 3:
            return None
        else:
            raise Exception("more than 3 cards in trick, that smells!")

    def start(self):
        # Wait for players
        while self._phase == GamePhase.WAITING:
            if len(self._player) == 3:
                self._phase = GamePhase.DEALING
        # card dealing
        self._deck.initialize_cards()
        self._deck.shuffle(seed=self._seed)
        for player in self._player:
            player.hand = self._deck.deal_cards()
        if not self._skip_bidding:
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
                stage_one_winner_bid = self._player[stage_one_winner].bid(
                    self._highest_bid
                )
                if stage_one_winner_bid > self._highest_bid:
                    self._highest_bid = stage_one_winner_bid
                    self._highest_bid_seat_id = stage_one_winner
                else:
                    break
        # take skat or not
        self._hand_game = not self._player[self._highest_bid_seat_id].pickup_skat()
        if not self._hand_game:
            _cards_in_skat = self._deck.deal_cards(2)
            if self._verbose:
                print(
                    f"p={self._highest_bid_seat_id} picks the skat: "
                    f"{_cards_in_skat}"
                )
            self._player[self._highest_bid_seat_id].receive_cards(_cards_in_skat)
            self._skat = self._player[self._highest_bid_seat_id].press_skat()
            if self._verbose:
                print(f"p={self._highest_bid_seat_id} puts {self._skat} in skat")
        else:
            if self._verbose:
                print(f"p={self._highest_bid_seat_id} discards the skat")
            self._skat = self._deck.deal_cards(2)
            if self._verbose:
                print(f"skat={self._skat}")
        for p in self._player:
            if self._verbose:
                print(p)
        if self._verbose:
            print(f"skat={self._skat}")
        # game declaration
        self._game = self._player[self._highest_bid_seat_id].declare_game()
        # add skat to tricks
        for card in self._skat:
            self._player[self._highest_bid_seat_id].trick_stack.append(card)
        # card_outplay
        while any(len(p.hand) for p in self._player):
            self._game.new_trick()
            while not self._game.trick.is_full:
                self._game.trick.append(
                    self.next_player,
                    self._player[self.next_player].play_card(self._game.trick),
                )
                if self._verbose:
                    print(f"trick={self._game.trick}")
            self._trick_history.append(copy.deepcopy(self._game.trick))
            self._player[self._game.trick.winner].take_trick(self._game.trick)
        # counting
        for p in self._player:
            is_soloist = p.seat_id == self._highest_bid_seat_id
            if self._verbose:
                print(
                    f"p={p.seat_id} {'*' if is_soloist else ' '} h={p.hand} points={p.trick_stack_value}"
                )
        # winner
        soloist_won = self.points_soloist > self.points_defenders
        if self._verbose:
            print(f"p={self._highest_bid_seat_id} {'won' if soloist_won else 'lost'}")
        if soloist_won:
            return self._highest_bid_seat_id, 1
        elif not soloist_won:
            return self._highest_bid_seat_id, -2
