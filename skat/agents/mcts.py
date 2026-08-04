from __future__ import annotations

from random import Random

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
    ) -> None:
        super().__init__(bidding=bidding)
        self.simulations = simulations
        self.determinizations = determinizations
        self.exploration = exploration
        self.rng = Random(seed)

    # ------------------------------------------------------------------ play

    def choose_card(self, valid_moves: set[Card]) -> Card:
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        choice = self.search(valid_moves)
        self.state.hand.remove(choice)
        return choice

    def search(self, valid_moves: set[Card]) -> Card:
        """Pick a card from `valid_moves` without mutating the hand."""
        moves = list(valid_moves)
        if len(moves) == 1:
            return moves[0]

        per_world = max(1, self.simulations // max(1, self.determinizations))
        totals: dict[Card, list[float]] = {move: [0, 0.0] for move in moves}
        for _ in range(self.determinizations):
            root = self.determinize()
            for move, (visits, value_sum) in uct_search(
                root, per_world, self.rng, self.exploration
            ).items():
                stats = totals.setdefault(move, [0, 0.0])
                stats[0] += visits
                stats[1] += value_sum

        def score(move: Card) -> float:
            visits, value_sum = totals[move]
            return value_sum / visits if visits else -1.0

        return max(moves, key=score)

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

        voids = infer_voids(round_.trick_history.buffer, game.trick)
        deal = deal_unseen(unseen, capacities, voids, self.rng)

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
        )

    # -------------------------------------------------------------- bidding

    def max_bid(self) -> int:
        """Pass outright on a hand not worth playing, whatever the style says."""
        assert self.state is not None
        if not playable(set(self.state.hand)):
            return 0
        return super().max_bid()
