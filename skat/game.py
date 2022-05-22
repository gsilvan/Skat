import copy
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import torch

from skat.agents.random import RandomAgent
from skat.card import Card
from skat.deck import Deck
from skat.games import Game
from skat.hand import Hand, HandOrder
from skat.player import Player
from skat.trick import TrickHistory
from skat.utils import disjoint


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
        verbose=False,
        seed=None,
        hold_position=False,
        declare_game=None,
    ) -> None:
        self.rounds = rounds
        self.agents = agents
        self.scores = [0, 0, 0]
        self.dealer = 0
        self.verbose = verbose
        self.seed = seed
        self.hold_position = hold_position
        self.declare_game = declare_game

    def start(self):
        for _ in range(self.rounds):
            r = Round(
                dealer=self.dealer,
                skip_bidding=True,
                solo_player_id=self.dealer,
                start=False,
                verbose=self.verbose,
                agents=self.agents,
                seed=self.seed,
                hand_game=True,
                declare_game=self.declare_game,
            )
            soloist, points = r.start()
            self.scores[soloist] += points
            self.scores[(soloist + 1) % 3] += -points / 2
            self.scores[(soloist + 2) % 3] += -points / 2
            if not self.hold_position:
                self.dealer = (self.dealer + 1) % 3


class Round:
    def __init__(
        self,
        dealer: int = 0,
        skip_bidding: bool = False,
        solo_player_id: int = -42,
        declare_game: Game = None,
        hand_game: Optional[bool] = None,
        agents: Optional[list] = None,
        deck: Deck = Deck(),
        start: bool = True,
        verbose: bool = False,
        seed=None,
        phase: GamePhase = GamePhase.WAITING,
        initial_cards: Optional[Union[list, tuple]] = None,
    ) -> None:
        if agents is None:
            agents = list()
        self.player: list[Player] = list()
        self.phase: GamePhase = phase
        self.dealer: int = dealer
        self.highest_bid: int = 0
        self.solo_player_id: int = solo_player_id
        self.deck: Deck = deck
        self.deal_deck: bool = True
        self.initial_cards = initial_cards
        self.verbose = verbose
        self.hand_game = hand_game
        self.skat: list[Card] = list()
        self.game: Optional[Game] = declare_game
        self.skip_bidding: bool = skip_bidding
        self.trick_history = TrickHistory()
        self.seed = seed
        if len(agents) > 0 and len(agents) != 3:
            raise Exception("specify either 3 players or None")
        elif len(agents) == 0:
            # init default agents
            self.init_players()
        elif len(agents) == 3:
            self.init_players(agents)
        if self.initial_cards:
            if len(self.initial_cards) == 4:
                # 4 seems good, but are 0,1,2,3 disjoint?
                if not disjoint(self.initial_cards):
                    raise Exception("initial cards are not disjoint!")
            else:
                raise Exception("provide 4 items, initial_cards=(p0, p1, p2, skat)")
            self.deal_deck = False
        if start:
            # start the game with selected features immediately
            self.start()

    @property
    def front_hand(self) -> int:
        """Return player_id in front-hand-position"""
        if (
            len(self.trick_history) > 0
            and self.trick_history.buffer[-1].winner is not None
        ):
            return self.trick_history.buffer[-1].winner
        else:
            return (self.dealer + 1) % 3

    @property
    def middle_hand(self) -> int:
        """Return player_id in middle-hand-position"""
        return (self.front_hand + 1) % 3

    @property
    def back_hand(self) -> int:
        """Return player_id in back-hand-position"""
        return (self.front_hand + 2) % 3

    @property
    def points_soloist(self) -> int:
        """Return solo player's points."""
        soloist = self.solo_player_id
        if soloist == -42:
            return 0
        return self.player[soloist].trick_stack_value

    @property
    def points_defenders(self) -> int:
        """Return accumulated defender's points."""
        soloist = self.solo_player_id
        if soloist == -42:
            return 0
        return (
            self.player[(soloist + 1) % 3].trick_stack_value
            + self.player[(soloist + 2) % 3].trick_stack_value
        )

    def init_players(self, agents=None) -> None:
        if not agents:
            for i in range(3):
                self.player.append(Player(RandomAgent(), i))
        else:
            for idx, agent in enumerate(agents):
                self.player.append(Player(agent, idx))
        for player in self.player:
            player.set_state(self)

    @property
    def next_player(self) -> int:
        """Return player_id of the next player who has to act."""
        if self.game is None:
            # if the game is not declared, there is no trick to look up.
            return self.front_hand
        match len(self.game.trick):
            case 0:
                return self.front_hand
            case 1:
                return self.middle_hand
            case 2:
                return self.back_hand
            case 3:
                return self.front_hand
            case _:
                raise Exception("more than 3 cards in trick, that smells!")

    @property
    def soloist_leading(self) -> bool:
        """Return True if soloist is leading."""
        return self.points_soloist > self.points_defenders

    @property
    def defenders_leading(self) -> bool:
        """Return True if defenders are leading."""
        return self.points_defenders > self.points_soloist

    @property
    def soloist_won(self) -> bool:
        """Return True if the soloist has (already) won the game."""
        return self.points_soloist > 60

    @property
    def soloist_won_schneider(self) -> bool:
        """Return True if the soloist has (already) archived schneider upgrade."""
        return self.points_soloist > 90

    @property
    def soloist_won_schwarz(self) -> bool:
        """Return True if the soloist has won all trick points."""
        return self.points_soloist == 120

    @property
    def defenders_won(self) -> bool:
        """Return True if the defenders have (already) beaten the soloist."""
        return self.points_defenders >= 60

    @property
    def defenders_won_schneider(self) -> bool:
        """Return True if the defenders have (already) archived schneider upgrade."""
        return self.points_defenders >= 90

    @property
    def defenders_won_schwarz(self) -> bool:
        """Return True if the defenders have won all trick points."""
        return self.points_defenders == 120

    @property
    def is_finished(self) -> bool:
        """Return True if the round is finished."""
        return not any(len(p.hand) for p in self.player)

    def deal(self) -> None:
        """Deal cards."""
        if self.deal_deck:
            self.deck.initialize_cards()
            self.deck.shuffle(seed=self.seed)
            for player in self.player:
                player.hand = Hand(self.deck.deal_cards())
                player.hand.sort(order=HandOrder("♦♥♠♣", "789QKXA", "♦♥♠♣", "J"))
        else:
            for idx, player in enumerate(self.player):
                player.hand = Hand(self.initial_cards[idx])  # type: ignore
                player.hand.sort(order=HandOrder("♦♥♠♣", "789QKXA", "♦♥♠♣", "J"))
            self.skat = self.initial_cards[3]  # type: ignore

    def get_state(self, player_id) -> np.ndarray:
        """Return a state vector for a given player. A player has a limited view on the
        state in incomplete information games."""
        __hand = self.player[player_id].hand.as_vector
        __color = self.game.to_numpy()  # type: ignore
        __points = np.array(
            [self.points_soloist / 120.0, self.points_defenders / 120.0]
        )
        __played_cards = np.array([])
        for i in range(3):
            __played_cards = np.concatenate(
                (__played_cards, self.trick_history.to_numpy(player_id=i))
            )
        __trick_value = np.array([self.game.trick.value / 120.0])  # type: ignore
        __front_hand = np.zeros(3, dtype=int)
        if player_id == self.front_hand:
            __front_hand[0] = 1
        elif player_id == self.middle_hand:
            __front_hand[1] = 1
        elif player_id == self.back_hand:
            __front_hand[2] = 1
        # assert len(__hand) == 32
        # assert len(__color) == 5
        # assert len(__points) == 2
        # assert len(__trick_value) == 1
        # assert len(__played_cards) == 96
        return np.concatenate(
            (__hand, __color, __points, __played_cards, __trick_value, __front_hand),
            dtype=np.float32,
        )

    def get_state_t(self, player_id) -> torch.Tensor:
        state = self.get_state(player_id)
        return torch.tensor(state).unsqueeze(0)

    def step(self) -> bool:
        """Do a step. A step is one single action of one player."""
        if self.phase == GamePhase.PLAYING:
            if self.game is None:
                raise Exception("ur doin it worng")
            self.game.trick.append(
                self.next_player,
                self.player[self.next_player].play_card(self.game.trick),
            )
            if self.verbose:
                print(f"trick={self.game.trick}")
            if self.game.trick.is_full:
                assert self.game.trick.winner is not None  # type safety
                self.trick_history.append(copy.deepcopy(self.game.trick))
                self.player[self.game.trick.winner].take_trick(self.game.trick)
                self.game.new_trick()
                # ping all player that the trick is done
                for p in self.player:
                    is_terminal = len(self.trick_history) == 10
                    p.strategy.trick_done_event(is_terminal)
            return True
        return False

    def step_player(self, player_id) -> tuple[int, bool]:
        """
        Perform a step for player_id and do all steps other players steps, until it is
        player_id's turn again.
        """
        old_points = self.player[player_id].trick_stack_value
        while True:
            step = self.step()
            if not step:
                # if no step was done, the game is in terminal state
                return 0, self.is_finished
            if self.next_player == player_id:
                current_points = self.player[player_id].trick_stack_value
                reward = current_points - old_points
                return reward, self.is_finished

    def bidding(self) -> None:
        while True:
            middle_hand_bid = self.player[self.middle_hand].bid(self.highest_bid)
            if middle_hand_bid > self.highest_bid:
                self.highest_bid = middle_hand_bid
                self.solo_player_id = self.middle_hand
            else:
                break
            front_hand_bid = self.player[self.front_hand].bid(self.highest_bid)
            if front_hand_bid > self.highest_bid:
                self.highest_bid = front_hand_bid
                self.solo_player_id = self.front_hand
            else:
                break
        stage_one_winner = self.solo_player_id
        while True:
            back_hand_bid = self.player[self.back_hand].bid(self.highest_bid)
            if back_hand_bid > self.highest_bid:
                self.highest_bid = back_hand_bid
                self.solo_player_id = self.back_hand
            else:
                break
            stage_one_winner_bid = self.player[stage_one_winner].bid(self.highest_bid)
            if stage_one_winner_bid > self.highest_bid:
                self.highest_bid = stage_one_winner_bid
                self.solo_player_id = stage_one_winner
            else:
                break

    def start(self):
        # Wait for players
        while self.phase == GamePhase.WAITING:
            if len(self.player) == 3:
                self.phase = GamePhase.DEALING
        # card dealing
        self.deal()
        if not self.skip_bidding:
            self.phase = GamePhase.BIDDING
            # game bidding
            self.bidding()
        # take skat or not
        if self.hand_game is None:
            # ask player
            self.hand_game = not self.player[self.solo_player_id].pickup_skat()
        if not self.hand_game:
            _cards_in_skat = self.deck.deal_cards(2)
            if self.verbose:
                print(f"p={self.solo_player_id} picks the skat: " f"{_cards_in_skat}")
            self.player[self.solo_player_id].receive_cards(_cards_in_skat)
            self.skat = self.player[self.solo_player_id].press_skat()
            if self.verbose:
                print(f"p={self.solo_player_id} puts {self.skat} in skat")
        else:
            if self.verbose:
                print(f"p={self.solo_player_id} discards the skat")
            if self.deal_deck:
                self.skat = self.deck.deal_cards(2)
            if self.verbose:
                print(f"skat={self.skat}")
        for p in self.player:
            if self.verbose:
                print(p)
        if self.verbose:
            print(f"skat={self.skat}")
        # game declaration
        if self.game is None:
            self.game = self.player[self.solo_player_id].declare_game()
        # add skat to tricks
        for card in self.skat:
            self.player[self.solo_player_id].trick_stack.append(card)
        # card_outplay
        self.phase = GamePhase.PLAYING
        while not self.is_finished:
            self.step()
        # counting
        self.phase = GamePhase.COUNTING
        for p in self.player:
            is_soloist = p.seat_id == self.solo_player_id
            if self.verbose:
                print(
                    f"p={p.seat_id} {'*' if is_soloist else ' '} h={p.hand} points={p.trick_stack_value}"
                )
        # winner
        soloist_won = self.points_soloist > self.points_defenders
        if self.verbose:
            print(f"p={self.solo_player_id} {'won' if soloist_won else 'lost'}")
        if soloist_won:
            return self.solo_player_id, 1
        elif not soloist_won:
            return self.solo_player_id, -2
