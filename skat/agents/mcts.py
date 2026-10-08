from __future__ import annotations

from skat.agents import Agent
from skat.agents.bidding import BiddingStyle, playable
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.search import (
    DEFAULT_EXPLORATION,
    SKAT_SLOT,
    PlayoutState,
    deal_unseen,
    infer_voids,
    uct_search,
)
from skat.agents.search.playout import copy_trick
from skat.card import Card
from skat.games.null import Null
from skat.hand import FULL_HAND
from skat.rewards import MCTS_TIEBREAK_OBJECTIVE, REWARD_OBJECTIVE, victory_status


class MoveTotals(dict[Card, list[float]]):
    """Primary totals plus independent bonus sums over all sampled worlds."""

    def __init__(self, moves):
        super().__init__((move, [0, 0.0]) for move in moves)
        self.bonus_sums = dict.fromkeys(moves, 0.0)


class MCTSAgent(HeuristicDeclarer, Agent):
    """
    Monte-Carlo tree search over determinized worlds (PIMC / determinized UCT).

    Skat is an imperfect-information game, so plain UCT has nothing to search:
    the agent does not know the other two hands. Instead each move samples a
    number of *determinizations* — complete deals consistent with everything
    the agent has seen (its own cards, every card played so far, and which
    suits opponents have shown themselves void in) — runs an independent UCT
    tree on each resulting perfect-information game, and plays the move with
    the best average value across all worlds.

    Only card play is searched. Bidding, skat handling and game declaration use
    the small documented heuristics at the bottom of this module; searching them
    would mean evaluating a whole round per candidate and is left for later.
    """

    def __init__(
        self,
        simulations: int = 600,
        determinizations: int = 20,
        exploration: float = DEFAULT_EXPLORATION,
        seed=None,
        bidding: BiddingStyle = BiddingStyle.CONSERVATIVE,
        bonus_tiebreak: bool = True,
    ) -> None:
        super().__init__(bidding=bidding, seed=seed)
        self.simulations = simulations
        self.determinizations = determinizations
        self.exploration = exploration
        self.bonus_tiebreak = bonus_tiebreak

    # ------------------------------------------------------------------ play

    def choose_card(self, valid_moves: set[Card]) -> Card:
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        choice = self.search(valid_moves)
        self.state.hand.remove(choice)
        return choice

    def search(self, valid_moves: set[Card]) -> Card:
        """Pick a card from `valid_moves` without mutating the hand."""
        moves = sorted(valid_moves, key=lambda card: card.np_index)
        if len(moves) == 1:
            return moves[0]
        return self.best(moves, self.tally(moves))

    def tally(self, moves: list[Card]) -> dict[Card, list[float]]:
        """
        Run the search and report `{move: [visits, value_sum]}`, summed over
        every determinization.

        Split out from `search` so `RecordingMCTSAgent` can write the numbers
        down on the way past: they are what a distilled policy learns from, and
        the base agent throws them away as soon as it has picked a card.
        """
        per_world = max(1, self.simulations // max(1, self.determinizations))
        totals = MoveTotals(moves)
        for _ in range(self.determinizations):
            root = self.determinize()
            result = uct_search(
                root,
                per_world,
                self.rng,
                self.exploration,
                bonus_tiebreak=self.bonus_tiebreak,
            )
            for move, (visits, value_sum) in result.items():
                stats = totals.setdefault(move, [0, 0.0])
                stats[0] += visits
                stats[1] += value_sum
                if self.bonus_tiebreak:
                    totals.bonus_sums[move] += result.bonus_sums[move]
        return totals

    @staticmethod
    def best(moves: list[Card], totals: dict[Card, list[float]]) -> Card:
        """The move with the best average value. An unvisited move scores -1."""

        def score(move: Card) -> tuple[float, float]:
            visits, value_sum = totals[move]
            bonus_sum = getattr(totals, "bonus_sums", {}).get(move, 0.0)
            return (value_sum / visits, bonus_sum / visits) if visits else (-1.0, -1.0)

        return max(moves, key=score)

    def deal(self, unseen, capacities, voids) -> dict[int, list[Card]]:
        """
        Sample one assignment of the unseen cards over the seats and the skat.

        Uniform over everything the constraints allow. Split out from
        `determinize` as the seam `skat.agents.belief_mcts.BeliefMCTSAgent`
        replaces to sample from a learned belief instead - which is the only
        thing that agent changes, so any difference between the two is
        attributable to the sampling alone.
        """
        return deal_unseen(unseen, capacities, voids, self.rng)

    def determinize(self) -> PlayoutState:
        """
        Sample one perfect-information world consistent with what we've seen,
        as a `PlayoutState` positioned at our own turn.
        """
        assert self.state is not None
        me = self.state.seat_id
        round_ = self.state.public_state
        assert round_ is not None and round_.game is not None
        game = round_.game
        soloist = round_.solo_player_id

        # Everything the completed tricks tell us: who has how many points, who
        # has already taken a trick (all that matters in null), what's gone.
        points = [0, 0, 0]
        soloist_took_trick = False
        played: set[Card] = set()
        for trick in round_.trick_history.buffer:
            winner = trick.winner
            assert winner is not None
            points[winner] += trick.value
            soloist_took_trick |= winner == soloist
            played.update(card for _, card in trick.buffer)
        played.update(card for _, card in game.trick.buffer)

        my_hand = list(self.state.hand)
        # We only know the skat if we are the soloist and actually picked it up.
        known_skat = list(round_.skat) if me == soloist and not round_.hand_game else []

        seen = played | set(my_hand) | set(known_skat)
        unseen = [card for card in FULL_HAND if card not in seen]

        capacities = {
            seat: len(round_.player[seat].hand) for seat in range(3) if seat != me
        }
        skat_size = len(unseen) - sum(capacities.values())
        if skat_size:
            capacities[SKAT_SLOT] = skat_size

        # The live trick counts too. If the seat before me could not follow a
        # moment ago, that is the freshest and most relevant void evidence
        # there is - and passing only the completed tricks threw away up to two
        # cards of it at every single decision.
        voids = infer_voids([*round_.trick_history.buffer, game.trick], game.trick)
        deal = self.deal(unseen, capacities, voids)

        hands: list[list[Card]] = [[], [], []]
        hands[me] = my_hand
        for seat, cards in deal.items():
            if seat != SKAT_SLOT:
                hands[seat] = cards

        # The skat's points are booked to the soloist at the start of play, so
        # they belong in the totals — known if we pressed them, guessed if not.
        skat = known_skat or deal.get(SKAT_SLOT, [])
        points[soloist] += sum(card.value for card in skat)

        return PlayoutState(
            trick=copy_trick(game.trick),
            hands=hands,
            to_move=me,
            points=points,
            soloist=soloist,
            is_null=isinstance(game, Null),
            soloist_took_trick=soloist_took_trick,
            overbid=round_.overbid,
            secured_winner=(
                0
                if isinstance(game, Null)
                else victory_status(
                    round_.points_soloist, round_.points_defenders, round_.overbid
                )
            ),
        )

    # -------------------------------------------------------------- bidding

    def max_bid(self) -> int:
        """Pass outright on a hand not worth playing, whatever the style says."""
        assert self.state is not None
        # NEURAL replaces this hand-written play/pass gate. Keeping both would
        # censor exactly the weak-looking wins the model was introduced to find.
        learned_or_searched = {BiddingStyle.NEURAL, BiddingStyle.MCTS}
        if self.bidding not in learned_or_searched and not playable(
            set(self.state.hand)
        ):
            return 0
        return super().max_bid()


class RecordingMCTSAgent(MCTSAgent):
    """
    An `MCTSAgent` that writes down what its search believed, as training data
    for `skat.agents.distill`.

    Play is identical to the base agent given the same seed - the only thing
    added is the write. It lives here rather than in `skat.agents.distill`
    because it is the one piece of the distillation that has to know about
    `Player` and `Round`, the same way `skat.agents.dqn` sits above the
    `Round`-free `skat.agents.rl`.

    `recorder` is a `skat.agents.distill.ShardWriter` in practice, but only its
    `new_round()` and `add()` are used.
    """

    def __init__(self, recorder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder
        if hasattr(self.recorder, "provenance"):
            self.recorder.provenance["reward_objective"] = (
                MCTS_TIEBREAK_OBJECTIVE if self.bonus_tiebreak else REWARD_OBJECTIVE
            )

    def new_round_event(self) -> None:
        super().new_round_event()
        # Groups the round's samples so the train/validation split can cut on
        # round boundaries. All three agents fire this before a card is played,
        # so the counter jumps by three per round; only the cuts matter.
        self.recorder.new_round()

    def search(self, valid_moves: set[Card]) -> Card:
        moves = sorted(valid_moves, key=lambda card: card.np_index)
        if len(moves) == 1:
            # Forced, and never searched. A masked softmax over a single legal
            # card has probability 1 and gradient 0, so there is nothing here
            # to learn - `PolicyAgent.select` short-circuits the same way, so
            # not recording it costs no train/serve skew either.
            return moves[0]
        totals = self.tally(moves)
        choice = self.best(moves, totals)
        assert self.state is not None and self.state.public_state is not None
        # Before `choose_card` takes the card out of the hand, so the recorded
        # observation is the one the student will be looking at.
        self.recorder.add(
            state=self.state.public_state.get_state(player_id=self.state.seat_id),
            totals=totals,
            moves=moves,
            choice=choice,
        )
        return choice
