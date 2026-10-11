"""Round lifecycle, validated actions, and reproducible tournaments."""

import copy
from dataclasses import asdict, dataclass, replace
from enum import Enum, auto
from random import Random

import numpy as np
from tqdm import tqdm

from skat.agents.bidding import VALID_BIDS
from skat.agents.random import RandomAgent
from skat.card import Card
from skat.deck import Deck
from skat.games import Game
from skat.hand import FULL_HAND, Hand, HandOrder
from skat.player import Player
from skat.rules import (
    Claim,
    Concede,
    GameDeclaration,
    OpenHand,
    RoundResult,
    can_soloist_take_trick,
    evaluate_game,
    null_overbid_substitute,
)
from skat.trick import Trick, TrickHistory


class GamePhase(Enum):
    WAITING = auto()
    DEALING = auto()
    BIDDING = auto()
    DECLARING = auto()
    PLAYING = auto()
    COUNTING = auto()
    EINGEPASST = auto()


@dataclass(frozen=True)
class PlayerView:
    seat_id: int
    dealer: int
    hand: tuple[Card, ...]
    hand_sizes: tuple[int, ...]
    solo_player_id: int
    highest_bid: int
    hand_game: bool | None
    skat: tuple[Card, ...]  # only known to a soloist who picked it up
    game: Game | None
    trick_history: tuple[Trick, ...]
    points: tuple[int, ...]  # publicly won trick points, no hidden skat
    trick_counts: tuple[int, ...]
    exposed_hands: tuple[tuple[Card, ...], ...]
    observation: np.ndarray | None


@dataclass
class SeatScore:
    solo_rounds: int = 0
    solo_wins: int = 0
    defence_rounds: int = 0
    defence_wins: int = 0
    overbids: int = 0
    card_points: int = 0
    game_points: int = 0

    @property
    def rounds(self):
        return self.solo_rounds + self.defence_rounds

    @property
    def wins(self):
        return self.solo_wins + self.defence_wins


class Tournament:
    def __init__(
        self,
        rounds=32,
        agents=None,
        verbose=False,
        seed=None,
        hold_position=False,
        declare_game=None,
        initial_cards=None,
        iss_file=None,
        progress=False,
    ):
        if rounds < 0:
            raise ValueError("rounds must be nonnegative")
        self.rounds = rounds
        self.progress = progress
        self.agents = (
            list(agents)
            if agents is not None
            else [RandomAgent(None if seed is None else seed + i) for i in range(3)]
        )
        if len(self.agents) != 3:
            raise ValueError("three agents required")
        self.scores = [0, 0]
        self.seat_scores = [SeatScore() for _ in range(3)]
        self.round_records = []
        self.passed = 0
        self.dealer = 0
        self.verbose = verbose
        self.seed = seed
        self.rng = Random(seed)
        self.hold_position = hold_position
        self.declare_game = declare_game
        self.initial_cards = initial_cards
        self.iss_games = None
        if iss_file:
            import iss

            self.iss_games = iss.ISSGames(file_name=iss_file)

    def next_seed(self):
        return None if self.seed is None else self.rng.getrandbits(64)

    def record(self, round_, soloist_won, round_index=None, seed=None):
        result = getattr(round_, "result", None)
        self.round_records.append(
            {
                "round": round_index,
                "seed": seed,
                "result": asdict(result) if result is not None else None,
            }
        )
        if soloist_won is None:
            self.passed += 1
            return
        self.scores[0 if soloist_won else 1] += 1
        for seat, score in enumerate(self.seat_scores):
            if seat == round_.solo_player_id:
                score.solo_rounds += 1
                score.solo_wins += int(soloist_won)
                score.overbids += int(round_.overbid)
                score.card_points += round_.points_soloist
                score.game_points += result.score if result else 0
            else:
                score.defence_rounds += 1
                score.defence_wins += int(not soloist_won)
                score.card_points += round_.points_defenders

    def summary(self):
        def rate(wins, rounds):
            return f"{wins}/{rounds} ({wins / rounds:.0%})" if rounds else "-"

        lines = [
            f"{self.rounds} rounds, {self.rounds - self.passed} played, {self.passed} passed out",
            f"soloists won {self.scores[0]}, defenders won {self.scores[1]}",
            "seat agent          as soloist    on defence    overbid game pts",
        ]
        for seat, score in enumerate(self.seat_scores):
            lines.append(
                f"{seat:4} {type(self.agents[seat]).__name__:14} "
                f"{rate(score.solo_wins, score.solo_rounds):14} "
                f"{rate(score.defence_wins, score.defence_rounds):14} "
                f"{score.overbids:7} {score.game_points:8}"
            )
        return "\n".join(lines)

    def start(self):
        samples = (
            self.iss_games.sample(n=self.rounds, seed=self.seed)
            if self.iss_games
            else None
        )
        with tqdm(
            total=self.rounds,
            desc="Runden",
            unit="Runde",
            dynamic_ncols=True,
            disable=not self.progress,
            postfix={"offen": self.rounds},
        ) as progress:
            for i in range(self.rounds):
                seed = self.next_seed()
                kwargs = {}
                if samples:
                    sample = samples[i]
                    # ISS hands have already been pressed; replay them without a second pickup.
                    game = sample.declaration.make_game()
                    kwargs = dict(
                        skip_bidding=True,
                        solo_player_id=sample.soloist,
                        hand_game=True,
                        replay=True,
                        declare_game=game,
                        initial_cards=sample.deck.to_list(),
                    )
                else:
                    kwargs = dict(
                        declare_game=copy.deepcopy(self.declare_game),
                        initial_cards=self.initial_cards,
                    )
                round_ = Round(
                    dealer=self.dealer,
                    start=False,
                    agents=self.agents,
                    seed=seed,
                    verbose=self.verbose,
                    **kwargs,
                )
                self.record(round_, round_.start(), i, seed)
                if self.verbose:
                    tqdm.write(f"round {i + 1}: {round_.result}")
                if not self.hold_position:
                    self.dealer = (self.dealer + 1) % 3
                progress.set_postfix(offen=self.rounds - i - 1, refresh=False)
                progress.update(1)
        return self.round_records


class Round:
    def __init__(
        self,
        dealer=0,
        skip_bidding=False,
        solo_player_id=-42,
        declare_game=None,
        hand_game=None,
        agents=None,
        deck=None,
        start=True,
        verbose=False,
        seed=None,
        phase=GamePhase.WAITING,
        initial_cards=None,
        replay=False,
    ):
        if dealer not in range(3) or solo_player_id not in (-42, 0, 1, 2):
            raise ValueError("invalid seat")
        if skip_bidding and solo_player_id == -42:
            raise ValueError("skipping bidding requires a soloist")
        self.dealer, self.solo_player_id = dealer, solo_player_id
        self.highest_bid = self.game_value = 0
        self.game = (
            declare_game.make_game()
            if isinstance(declare_game, GameDeclaration)
            else declare_game
        )
        self.hand_game = (
            self.game.declaration.hand
            if hand_game is None and self.game is not None
            else hand_game
        )
        self.phase, self.seed, self.verbose = phase, seed, verbose
        self.skip_bidding, self.replay = skip_bidding, replay
        self.deck = Deck() if deck is None else deck
        self.initial_cards = initial_cards
        self.deal_deck = initial_cards is None
        if initial_cards is not None:
            flat = [c for block in initial_cards for c in block]
            if (
                len(initial_cards) != 4
                or [len(x) for x in initial_cards] != [10, 10, 10, 2]
                or len(set(flat)) != 32
                or set(flat) != set(FULL_HAND)
            ):
                raise ValueError("initial_cards must partition the pack as 10/10/10/2")
        self.skat = []
        self.original_solo_cards = ()
        self.trick_history = TrickHistory()
        self.result = None
        self.exposed = set()
        self.awarded_tricks = [0, 0, 0]
        self._prepared = False
        agents = (
            list(agents)
            if agents
            else [RandomAgent(None if seed is None else seed + i) for i in range(3)]
        )
        if len(agents) != 3:
            raise ValueError("three agents required")
        self.player = [Player(agent, seat) for seat, agent in enumerate(agents)]
        self._refresh()
        if start:
            self.start()

    @property
    def front_hand(self):
        return (
            self.trick_history[-1].winner
            if len(self.trick_history)
            else (self.dealer + 1) % 3
        )

    @property
    def middle_hand(self):
        return (self.front_hand + 1) % 3

    @property
    def back_hand(self):
        return (self.front_hand + 2) % 3

    @property
    def next_player(self):
        return (self.front_hand + (len(self.game.trick) if self.game else 0)) % 3

    @property
    def points_soloist(self):
        return (
            self.player[self.solo_player_id].trick_stack_value
            if self.solo_player_id != -42
            else 0
        )

    @property
    def points_defenders(self):
        return (
            sum(
                p.trick_stack_value
                for p in self.player
                if p.seat_id != self.solo_player_id
            )
            if self.solo_player_id != -42
            else 0
        )

    @property
    def trick_counts(self):
        return tuple(
            sum(t.winner == seat for t in self.trick_history.buffer)
            + self.awarded_tricks[seat]
            for seat in range(3)
        )

    @property
    def party_tricks(self):
        solo = (
            self.trick_counts[self.solo_player_id] if self.solo_player_id != -42 else 0
        )
        return solo, sum(self.trick_counts) - solo

    @property
    def overbid(self):
        # A definitive result only exists at the end. Before then higher
        # earned grades may still cover a nominally expensive bid.
        return self.result.overbid if self.result else False

    @property
    def soloist_won(self):
        return self.result is not None and self.result.soloist_won is True

    @property
    def defenders_won(self):
        return self.result is not None and self.result.soloist_won is False

    @property
    def soloist_won_schneider(self):
        return (
            self.soloist_won
            and self.result.declaration.kind != "null"
            and self.points_defenders <= 30
        )

    @property
    def defenders_won_schneider(self):
        return (
            self.defenders_won
            and self.result.declaration.kind != "null"
            and self.points_soloist <= 30
        )

    @property
    def soloist_won_schwarz(self):
        return self.soloist_won and self.party_tricks == (10, 0)

    @property
    def defenders_won_schwarz(self):
        return self.defenders_won and self.party_tricks == (0, 10)

    @property
    def is_finished(self):
        return self.result is not None or (
            self._prepared and not any(len(p.hand) for p in self.player)
        )

    def is_solo_player(self, player_id):
        return player_id == self.solo_player_id

    def get_points(self, player_id, opponent_view=False):
        if opponent_view:
            return self.player[player_id].trick_stack_value
        return (
            self.points_soloist
            if self.is_solo_player(player_id)
            else self.points_defenders
        )

    @staticmethod
    def seats_from(player_id):
        return [(player_id + i) % 3 for i in range(3)]

    @property
    def skat_known(self):
        """Replay hands are already pressed; their contract preserves pickup knowledge."""
        return (
            (not self.game.declaration.hand)
            if self.replay and self.game is not None
            else self.hand_game is False
        )

    def view(self, player_id):
        visible_skat = (
            tuple(self.skat)
            if player_id == self.solo_player_id and self.skat_known
            else ()
        )
        points = tuple(
            sum(t.value for t in self.trick_history.buffer if t.winner == seat)
            for seat in range(3)
        )
        observation = self.get_state(player_id) if self.game is not None else None
        if observation is not None:
            observation.setflags(write=False)
        # Deepcopy prevents mutable Card/Trick/Game objects from exposing engine state.
        return copy.deepcopy(
            PlayerView(
                player_id,
                self.dealer,
                tuple(self.player[player_id].hand),
                tuple(len(p.hand) for p in self.player),
                self.solo_player_id,
                self.highest_bid,
                self.game.declaration.hand
                if self.replay and self.game is not None
                else self.hand_game,
                visible_skat,
                self.game,
                tuple(self.trick_history.buffer),
                points,
                self.trick_counts,
                tuple(
                    tuple(p.hand) if p.seat_id in self.exposed else ()
                    for p in self.player
                ),
                observation,
            )
        )

    def _refresh(self):
        for p in self.player:
            p.strategy.set_state(self.view(p.seat_id))

    def deal(self):
        if self.deal_deck:
            self.deck.initialize_cards()
            self.deck.shuffle(self.seed)
            for p in self.player:
                p.hand = Hand(self.deck.deal_cards())
            self.skat = self.deck.deal_cards(2)
        else:
            for p, cards in zip(self.player, self.initial_cards[:3]):
                p.hand = Hand(cards)
            self.skat = list(self.initial_cards[3])
        for p in self.player:
            p.hand.sort(HandOrder("♦♥♠♣", "789QKXA", "♦♥♠♣", "J"))
        self._refresh()

    def observer_points(self, player_id):
        counts = [
            sum(t.value for t in self.trick_history.buffer if t.winner == seat)
            for seat in range(3)
        ]
        solo = counts[self.solo_player_id] if self.solo_player_id != -42 else 0
        defence = sum(counts) - solo
        if player_id == self.solo_player_id and self.skat_known:
            solo += sum(c.value for c in self.skat)
        return (solo, defence) if player_id == self.solo_player_id else (defence, solo)

    def get_state(self, player_id):
        if self.game is None:
            raise ValueError("observation requires a declared game")
        seats = self.seats_from(player_id)
        solo_seat = np.array([int(s == self.solo_player_id) for s in seats])
        position = np.array(
            [
                int(player_id == s)
                for s in (self.front_hand, self.middle_hand, self.back_hand)
            ]
        )
        d = self.game.declaration
        # Existing 240-feature prefix is retained; public contract information follows.
        return np.concatenate(
            (
                self.player[player_id].hand.as_vector,
                self.game.to_numpy(),
                [int(player_id == self.solo_player_id)],
                solo_seat,
                np.array(self.observer_points(player_id)) / 120,
                *(self.trick_history.to_numpy(s) for s in seats),
                *(self.game.trick.to_numpy(s) for s in seats),
                position,
                [self.game.trick.value / 120],
                [
                    d.hand,
                    d.ouvert,
                    d.schneider_announced,
                    d.schwarz_announced,
                    self.highest_bid / 264,
                ],
                *(
                    Hand(self.player[s].hand).as_vector
                    if s in self.exposed
                    else np.zeros(32)
                    for s in seats
                ),
                [self.trick_counts[s] / 10 for s in seats],
                Hand(self.skat).as_vector
                if player_id == self.solo_player_id and self.skat_known
                else np.zeros(32),
            ),
            dtype=np.float32,
        )

    def get_state_t(self, player_id):
        import torch

        return torch.from_numpy(self.get_state(player_id)).unsqueeze(0)

    def cumulative_reward(self, player_id):
        if not self.result or self.result.soloist_won is None:
            return 0.0
        return (
            1.0
            if (player_id == self.solo_player_id) == self.result.soloist_won
            else -1.0
        )

    def _duel(self, sayer, hearer, value, hearer_holds=False):
        self._refresh()
        while True:
            offered = self.player[sayer].bid(value)
            if offered == 0:
                return (hearer if hearer_holds else None), value
            if offered not in VALID_BIDS or offered <= value:
                raise ValueError("bid must be a higher legal value or pass")
            response = self.player[hearer].bid(value, offered)
            if response not in (0, offered):
                raise ValueError("hearer must hold exactly the offer or pass")
            value = offered
            if response == 0:
                return sayer, value
            hearer_holds = True

    def bidding(self):
        winner, value = self._duel(self.middle_hand, self.front_hand, 0)
        if winner is None:
            winner, value = self._duel(self.back_hand, self.front_hand, 0)
        else:
            winner, value = self._duel(self.back_hand, winner, value, hearer_holds=True)
        if winner is None:
            claim = self.player[self.front_hand].bid(0)
            if claim:
                if claim not in VALID_BIDS:
                    raise ValueError("invalid opening bid")
                winner, value = self.front_hand, claim
        self.solo_player_id = winner if winner is not None else -42
        self.highest_bid = value
        self._refresh()

    def prepare(self):
        if self._prepared or self.result is not None:
            raise ValueError("round has already been prepared")
        for p in self.player:
            p.strategy.new_round_event()
        self.phase = GamePhase.DEALING
        self.deal()
        if not self.skip_bidding:
            self.phase = GamePhase.BIDDING
            self.bidding()
        if self.solo_player_id == -42:
            self.phase = GamePhase.EINGEPASST
            self.result = RoundResult(None, None, "passed")
            return
        self.phase = GamePhase.DECLARING
        solo = self.player[self.solo_player_id]
        if self.hand_game is None:
            self.hand_game = not solo.pickup_skat()
        if not self.hand_game:
            solo.receive_cards(self.skat)
            self._refresh()
            pressed = solo.press_skat()
            if (
                len(pressed) != 2
                or len(set(pressed)) != 2
                or not set(pressed) <= set(solo.hand)
            ):
                raise ValueError(
                    "press exactly two distinct cards from the twelve-card hand"
                )
            for c in pressed:
                solo.hand.remove(c)
            self.skat = list(pressed)
        self._refresh()
        if self.game is None:
            declared = solo.declare_game()
            self.game = (
                declared.make_game()
                if isinstance(declared, GameDeclaration)
                else declared
            )
        if not isinstance(self.game, Game):
            raise ValueError("agent must return a valid declaration")
        if not self.replay:
            # Hand is decided by the pickup action; cannot be claimed after pickup.
            if self.game.declaration.hand and not self.hand_game:
                raise ValueError("Hand cannot be declared after skat pickup")
            self.game.declaration = replace(
                self.game.declaration, hand=bool(self.hand_game)
            )
        self.original_solo_cards = tuple(solo.hand) + tuple(self.skat)
        solo.trick_stack.extend(self.skat)
        for p in self.player:
            p.hand.sort(self.game.order)
        if self.game.declaration.ouvert:
            self.exposed.add(self.solo_player_id)
        self._prepared = True
        self.phase = GamePhase.PLAYING
        self._refresh()

    def _finish(self, *, forced_winner=None, awarded_level=None, reason=None):
        d = self.game.declaration
        extra = {}
        if d.kind == "null" and self.highest_bid > d.base_value:
            value, substitute = null_overbid_substitute(
                self.original_solo_cards, self.highest_bid
            )
            extra = dict(null_overbid_value=value, null_overbid_declaration=substitute)
        self.result = evaluate_game(
            d,
            self.game.matadors(self.original_solo_cards),
            (self.points_soloist, self.points_defenders),
            self.party_tricks,
            self.highest_bid,
            soloist=self.solo_player_id,
            forced_winner=forced_winner,
            awarded_level=awarded_level,
            reason=reason,
            **extra,
        )
        self.game_value = self.result.game_value
        self.phase = GamePhase.COUNTING
        self._refresh()
        return self.result

    def play_card(self, player_id, card):
        if (
            self.phase is not GamePhase.PLAYING
            or self.result is not None
            or player_id != self.next_player
        ):
            raise ValueError("not this player's turn")
        p = self.player[player_id]
        if not isinstance(card, Card) or card not in p.valid_moves(self.game.trick):
            raise ValueError("card is absent or violates the follow-suit rule")
        p.hand.remove(card)
        self.game.trick.append(player_id, card)
        if self.game.trick.is_full:
            self.trick_history.append(copy.deepcopy(self.game.trick))
            self.player[self.game.trick.winner].take_trick(self.game.trick)
            self.game.new_trick()
            if not any(len(p.hand) for p in self.player):
                self._finish()
            elif self.game.declaration.kind == "null" and self.party_tricks[0] > 0:
                self._finish(awarded_level=0)
            self._refresh()
            for p in self.player:
                p.strategy.trick_done_event(self.is_finished)
        else:
            self._refresh()

    def concede(self, player_id, consenting_defenders=()):
        if self.phase is not GamePhase.PLAYING or player_id not in range(3):
            raise ValueError("concession requires an active game and a valid seat")
        defenders = set(range(3)) - {self.solo_player_id}
        if not set(consenting_defenders) <= defenders:
            raise ValueError("only defenders may consent")
        if (
            player_id == self.solo_player_id
            and len(self.player[player_id].hand) < 9
            and not consenting_defenders
        ):
            return False
        won = player_id != self.solo_player_id
        # Secured higher grades survive a concession; no speculative grades.
        winner_points = self.points_soloist if won else self.points_defenders
        level = int(winner_points >= 90)
        if won and self.game.declaration.kind != "null":
            # Defender concession grants a necessary higher grade when it is
            # still possible (ISkO 4.4.3 -> 4.1.5). Actual tricks/points rule
            # out grades already missed, and no speculative bonus is added.
            maximum = (
                2 if self.party_tricks[1] == 0 else int(self.points_defenders <= 30)
            )
            for candidate in range(level, maximum + 1):
                preview = evaluate_game(
                    self.game.declaration,
                    self.game.matadors(self.original_solo_cards),
                    (self.points_soloist, self.points_defenders),
                    self.party_tricks,
                    self.highest_bid,
                    forced_winner=True,
                    awarded_level=candidate,
                )
                if not preview.overbid:
                    level = candidate
                    break
        self._finish(forced_winner=won, awarded_level=level, reason="conceded")
        return True

    def claim(self, player_id, level=0, consenting_defenders=()):
        if self.phase is not GamePhase.PLAYING or player_id != self.solo_player_id:
            raise ValueError("only the soloist may claim during play")
        if level not in (0, 1, 2):
            raise ValueError("invalid claim level")
        defenders = set(range(3)) - {player_id}
        if not set(consenting_defenders) <= defenders:
            raise ValueError("only defenders may consent")
        d = self.game.declaration
        required = 2 if d.schwarz_announced else int(d.schneider_announced)
        if level < required or (d.kind == "null" and level):
            raise ValueError("claim must cover the announced objective")
        self.exposed.add(player_id)
        self._refresh()
        if set(consenting_defenders) != defenders:
            return False  # continue with publicly exposed cards
        self._finish(forced_winner=True, awarded_level=level, reason="claimed")
        return True

    def open_defender(self, player_id):
        if (
            self.phase is not GamePhase.PLAYING
            or player_id == self.solo_player_id
            or player_id not in range(3)
        ):
            raise ValueError("only a defender may open cards during play")
        possible = can_soloist_take_trick(
            self.game,
            [list(p.hand) for p in self.player],
            self.next_player,
            self.solo_player_id,
        )
        self.exposed.add(player_id)
        # A false opening awards all remaining tricks to the soloist; a valid
        # opening awards them to the defending party. Existing tricks survive.
        winner = self.solo_player_id if possible else player_id
        remaining = [c for p in self.player for c in p.hand] + [
            c for _, c in self.game.trick.buffer
        ]
        self.player[winner].trick_stack.extend(remaining)
        self.awarded_tricks[winner] += len(remaining) // 3
        for p in self.player:
            p.hand.clear()
        self.game.new_trick()
        return self._finish(reason="defender-open")

    def step(self):
        if self.phase is not GamePhase.PLAYING or self.is_finished:
            return False
        seat = self.next_player
        self._refresh()
        action = self.player[seat].play_card(self.game.trick)
        if isinstance(action, OpenHand):
            self.open_defender(seat)
        elif isinstance(action, Concede):
            consents = (
                [
                    p.seat_id
                    for p in self.player
                    if p.seat_id != self.solo_player_id
                    and p.strategy.confirm_concession()
                ]
                if seat == self.solo_player_id and len(self.player[seat].hand) < 9
                else []
            )
            self.concede(seat, consents)
        elif isinstance(action, Claim):
            if seat != self.solo_player_id:
                raise ValueError("only the soloist may claim")
            # Validate before exposing the hand or asking opponents.
            self.claim(seat, action.level)
            consents = [
                p.seat_id
                for p in self.player
                if p.seat_id != seat and p.strategy.confirm_claim(action.level)
            ]
            self.claim(seat, action.level, consents)
        else:
            self.play_card(seat, action)
        return True

    def step_player(self, player_id):
        old = self.get_points(player_id)
        while self.step():
            if self.is_finished or self.next_player == player_id:
                break
        return self.get_points(player_id) - old, self.is_finished

    def start(self):
        if not self._prepared and self.result is None:
            self.prepare()
        while not self.is_finished:
            self.step()
        return self.result.soloist_won
