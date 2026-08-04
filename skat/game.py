import copy
from enum import Enum, auto
from random import Random
from typing import Optional, Union

import numpy as np
import torch
import tqdm

import iss
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.deck import Deck
from skat.games import Game
from skat.hand import Hand, HandOrder
from skat.player import Player
from skat.trick import TrickHistory
from skat.utils.game_converter import get_game
from skat.utils.misc import disjoint


class GamePhase(Enum):
    WAITING = auto()
    DEALING = auto()
    BIDDING = auto()
    EINGEPASST = auto()
    DECLARING = auto()
    PLAYING = auto()
    COUNTING = auto()


def _rate(wins: int, rounds: int) -> str:
    """`wins/rounds (pct)`, or a dash if the seat never held that role."""
    if not rounds:
        return "-"
    return f"{wins}/{rounds} ({100 * wins // rounds}%)"


def _row(
    seat: str, agent: str, solo: str, defence: str, points: str
) -> str:
    """One line of the summary table."""
    return f"{seat:>4}  {agent:<14} {solo:>14} {defence:>14} {points:>9}"


class SeatScore:
    """
    What one seat achieved over a tournament. `card_points` accumulates the
    points of the party that seat played for, so the two defenders of a round
    are each credited with the full defence total.
    """

    def __init__(self) -> None:
        self.solo_rounds = 0
        self.solo_wins = 0
        self.defence_rounds = 0
        self.defence_wins = 0
        self.card_points = 0

    @property
    def rounds(self) -> int:
        return self.solo_rounds + self.defence_rounds

    @property
    def wins(self) -> int:
        return self.solo_wins + self.defence_wins


class Tournament:
    def __init__(
        self,
        rounds=32,
        agents=(RandomAgent(), RandomAgent(), RandomAgent()),
        verbose=False,
        seed=None,
        hold_position=False,
        declare_game=None,
        initial_cards=None,
        iss_file=None,
    ) -> None:
        self.rounds = rounds
        self.agents = agents
        #: [rounds won by the soloist, rounds won by the defence] - by role, so
        #: this says nothing about which agent played well. See `seat_scores`.
        self.scores = [0, 0]
        self.seat_scores = [SeatScore() for _ in range(3)]
        self.passed = 0  # rounds nobody bid on ("eingepasst")
        self.dealer = 0
        self.verbose = verbose
        self.seed = seed
        # One generator for the whole tournament, drawn from once per round, so
        # a seeded run stays reproducible while still dealing a fresh hand each
        # round. Reusing `seed` itself would deal the identical hand every time.
        self.round_seeds = Random(seed) if seed is not None else None
        self.hold_position = hold_position
        self.declare_game = declare_game
        self.initial_cards = initial_cards
        self.iss_games = None
        if iss_file:
            self.iss_games = iss.ISSGames(file_name=iss_file)

    def next_seed(self) -> Optional[int]:
        """A fresh per-round seed, or None when the tournament is unseeded."""
        if self.round_seeds is None:
            return None
        return self.round_seeds.randrange(2**32)

    def record(self, round_: "Round", soloist_won: Optional[bool]) -> None:
        """Fold one finished round into the running tallies."""
        if soloist_won is None:
            self.passed += 1
            return
        self.scores[0 if soloist_won else 1] += 1
        for seat, score in enumerate(self.seat_scores):
            if seat == round_.solo_player_id:
                score.solo_rounds += 1
                score.solo_wins += int(soloist_won)
                score.card_points += round_.points_soloist
            else:
                score.defence_rounds += 1
                score.defence_wins += int(not soloist_won)
                score.card_points += round_.points_defenders

    def summary(self) -> str:
        """Per-seat result table - who actually played well."""
        played = self.rounds - self.passed
        lines = [
            f"{self.rounds} rounds, {played} played, {self.passed} passed out",
            f"soloists won {self.scores[0]}, defenders won {self.scores[1]}",
            "",
            _row("seat", "agent", "as soloist", "on defence", "card pts"),
        ]
        for seat, score in enumerate(self.seat_scores):
            lines.append(
                _row(
                    str(seat),
                    type(self.agents[seat]).__name__,
                    _rate(score.solo_wins, score.solo_rounds),
                    _rate(score.defence_wins, score.defence_rounds),
                    str(score.card_points),
                )
            )
        return "\n".join(lines)

    def start(self):
        iss_sample = None
        if self.iss_games:
            iss_sample = self.iss_games.sample(n=self.rounds, seed=self.seed)

        print(f"starting tournament with {self.rounds}...")
        for i in tqdm.tqdm(range(self.rounds)):
            if self.iss_games:
                # ISS records already fix who bid what and won - replay as-is.
                self.initial_cards = iss_sample[i].deck.to_list()
                r = Round(
                    dealer=0,
                    skip_bidding=True,
                    solo_player_id=iss_sample[i].soloist,
                    start=False,
                    verbose=self.verbose,
                    agents=self.agents,
                    seed=self.next_seed(),
                    hand_game=True,
                    declare_game=get_game(iss_sample[i].type),
                    initial_cards=self.initial_cards,
                )
            else:
                # let the actual Sagen/Hören bidding determine the soloist.
                r = Round(
                    dealer=self.dealer,
                    skip_bidding=False,
                    start=False,
                    verbose=self.verbose,
                    agents=self.agents,
                    seed=self.next_seed(),
                    declare_game=self.declare_game,
                    initial_cards=self.initial_cards,
                )
            self.record(r, r.start())
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
        deck: Optional[Deck] = None,
        start: bool = True,
        verbose: bool = False,
        seed=None,
        phase: GamePhase = GamePhase.WAITING,
        initial_cards: Optional[Union[list, tuple]] = None,
    ) -> None:
        if agents is None:
            agents = list()
        if deck is None:
            deck = Deck()
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

    def is_solo_player(self, player_id) -> bool:
        return self.solo_player_id == player_id

    def get_points(self, player_id, opponent_view=False) -> int:
        if not opponent_view:
            if self.is_solo_player(player_id):
                return self.player[player_id].trick_stack_value
            else:
                return (
                    self.player[(player_id + 1) % 3].trick_stack_value
                    + self.player[(player_id + 2) % 3].trick_stack_value
                )
        return self.player[player_id].trick_stack_value

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
        __current_trick = self.game.trick.as_vector
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
            (
                __hand,
                __color,
                __points,
                __played_cards,
                __trick_value,
                __current_trick,
                __front_hand,
            ),
            dtype=np.float32,
        )

    def get_state_t(self, player_id) -> torch.Tensor:
        state = self.get_state(player_id)
        return torch.tensor(state, device="cpu").unsqueeze(0)
        # TODO: device should be derived from a config file

    def cumulative_reward(self, player_id) -> float:
        is_solo = player_id == self.solo_player_id

        reward_goals = {
            "points": 2.0 / 120.0,
            "win": 5.0,
            "schneider": 2.0,
            "schwarz": 2.0,
        }

        reward = 0.0

        if is_solo:
            reward += self.points_soloist * reward_goals["points"]
            reward -= self.points_defenders * reward_goals["points"]
            if self.soloist_won:
                reward += reward_goals["win"]
            if self.soloist_won_schneider:
                reward += reward_goals["schneider"]
            if self.soloist_won_schwarz:
                reward += reward_goals["schwarz"]
            if self.defenders_won:
                reward -= reward_goals["win"]
            if self.defenders_won_schneider:
                reward -= reward_goals["schneider"]
            if self.defenders_won_schwarz:
                reward -= reward_goals["schwarz"]
        else:
            reward -= self.points_soloist * reward_goals["points"]
            reward += self.points_defenders * reward_goals["points"]
            if self.soloist_won:
                reward -= reward_goals["win"]
            if self.soloist_won_schneider:
                reward -= reward_goals["schneider"]
            if self.soloist_won_schwarz:
                reward -= reward_goals["schwarz"]
            if self.defenders_won:
                reward += reward_goals["win"]
            if self.defenders_won_schneider:
                reward += reward_goals["schneider"]
            if self.defenders_won_schwarz:
                reward += reward_goals["schwarz"]
        return reward / 10.0

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

    def _duel(
        self,
        sayer_id: int,
        hearer_id: int,
        current_value: int,
        hearer_holds=False,
    ) -> tuple[Optional[int], int]:
        """
        One Sagen/Hören exchange (ISkO §3.3.2): `sayer_id` states ascending bid
        values; `hearer_id` either holds (accepts) each exact value or passes.
        Returns `(winner_id, value)`. `winner_id` is `None` if the sayer never
        said anything the hearer had to react to and `hearer_holds` was False
        (i.e. nobody in this duel has claimed a value yet).
        """
        while True:
            said = self.player[sayer_id].bid(current_value)
            if not said or said <= current_value:
                break
            held = self.player[hearer_id].bid(current_value, offer=said)
            if held == said:
                current_value = said
                hearer_holds = True
            else:
                # hearer passed on `said` -> sayer's claim stands
                return sayer_id, said
        if hearer_holds:
            return hearer_id, current_value
        return None, 0

    def bidding(self) -> None:
        """
        Determine the soloist via the official Sagen/Hören procedure (ISkO
        §3.3): Mittelhand states ascending values against Vorhand (§3.3.2-3.3.4);
        the winner of that duel then faces Hinterhand, who always says
        (§3.3.3-3.3.5). If nobody ever bids, Vorhand may still claim the skat or
        a hand game unprompted (§3.3.6). If she declines too, nothing was bid at
        all — the round is "eingepasst" (§3.3.7) and `solo_player_id` is left at
        its unset sentinel for `start()` to handle.
        """
        winner, value = self._duel(self.middle_hand, self.front_hand, 0)
        if winner is None:
            winner, value = self._duel(self.back_hand, self.front_hand, 0)
        else:
            winner, value = self._duel(self.back_hand, winner, value, hearer_holds=True)
        if winner is None:
            claim = self.player[self.front_hand].bid(0)
            if claim:
                winner, value = self.front_hand, claim
        self.highest_bid = value
        if winner is not None:
            self.solo_player_id = winner

    def start(self) -> Optional[bool]:
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
            if self.solo_player_id == -42:
                # nobody bid anything -> "eingepasst" (ISkO §3.2.19, §3.3.7),
                # a valid round outcome, not an error.
                self.phase = GamePhase.EINGEPASST
                return None
        # take skat or not
        if self.hand_game is None:
            # ask player
            self.hand_game = not self.player[self.solo_player_id].pickup_skat()
        if not self.hand_game:
            _cards_in_skat = self.deck.deal_cards(2)
            if self.verbose:
                print(f"p={self.solo_player_id} picks the skat: {_cards_in_skat}")
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
            # return self.solo_player_id, 1
            return True
        elif not soloist_won:
            # return self.solo_player_id, -2
            return False
