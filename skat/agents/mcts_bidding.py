"""Counterfactual MCTS game selection and play/pass bidding."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from skat.agents.declaring import press_for_game
from skat.agents.search import DEFAULT_EXPLORATION, SKAT_SLOT, PlayoutState, deal_unseen
from skat.agents.search.playout import copy_trick
from skat.agents.search.uct import uct_search
from skat.card import Card
from skat.games import Game
from skat.games.grand import Grand
from skat.games.suit import SuitGame
from skat.hand import FULL_HAND


def game_key(game: Game) -> str:
    """Stable, human-readable identifier for a bidding candidate."""
    if isinstance(game, SuitGame):
        return ("diamonds", "hearts", "spades", "clubs")[game.suit]
    return "grand"


def candidate_games() -> list[Game]:
    """The five point games the bidder evaluates; Null is intentionally out."""
    return [SuitGame(suit) for suit in range(4)] + [Grand()]


@dataclass(frozen=True)
class MCTSBidResult:
    game: Game
    candidate_probabilities: dict[str, float]
    confirmation_probability: float


@dataclass(frozen=True)
class _World:
    hands: tuple[tuple[Card, ...], tuple[Card, ...], tuple[Card, ...]]
    skat: tuple[Card, Card]


class MCTSBidder:
    """
    Pick a game and estimate whether it wins often enough to enter the auction.

    Each unknown world assigns the other twenty cards to the defenders and two
    cards to the skat. For every candidate the soloist takes that skat, presses
    two cards for the fixed game and searches the resulting perfect-information
    position with UCT. Candidate games share the same worlds so their first-stage
    comparison is paired. The winner is then evaluated again on fresh worlds;
    only that independent confirmation is compared with `threshold`.
    """

    def __init__(
        self,
        simulations: int = 600,
        determinizations: int = 20,
        confirmation_simulations: int = 1_200,
        confirmation_determinizations: int = 40,
        threshold: float = 0.65,
        exploration: float = DEFAULT_EXPLORATION,
        seed: int | None = None,
    ) -> None:
        positive = {
            "simulations": simulations,
            "determinizations": determinizations,
            "confirmation_simulations": confirmation_simulations,
            "confirmation_determinizations": confirmation_determinizations,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                f"MCTS bidding values must be positive: {', '.join(invalid)}"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("MCTS bidding threshold must be between 0 and 1")
        if exploration < 0.0:
            raise ValueError("MCTS bidding exploration must be non-negative")
        self.simulations = simulations
        self.determinizations = determinizations
        self.confirmation_simulations = confirmation_simulations
        self.confirmation_determinizations = confirmation_determinizations
        self.threshold = threshold
        self.exploration = exploration
        self.rng = Random(seed)

    def evaluate(self, cards: set[Card], seat: int, dealer: int) -> MCTSBidResult:
        if len(cards) != 10:
            raise ValueError(f"MCTS bidding expects ten cards, got {len(cards)}")
        if seat not in range(3) or dealer not in range(3):
            raise ValueError("seat and dealer must be in range 0..2")

        selection_worlds = self._sample_worlds(cards, seat, self.determinizations)
        games = candidate_games()
        probabilities = {
            game_key(game): self._estimate(
                cards, seat, dealer, game, selection_worlds, self.simulations
            )
            for game in games
        }
        best = max(
            games,
            key=lambda game: (
                probabilities[game_key(game)],
                game.value,
                -1 if game.suit is None else game.suit,
            ),
        )

        confirmation_worlds = self._sample_worlds(
            cards, seat, self.confirmation_determinizations
        )
        confirmation = self._estimate(
            cards,
            seat,
            dealer,
            best,
            confirmation_worlds,
            self.confirmation_simulations,
        )
        return MCTSBidResult(best, probabilities, confirmation)

    def _sample_worlds(
        self, cards: set[Card], soloist: int, count: int
    ) -> list[_World]:
        unseen = [card for card in FULL_HAND if card not in cards]
        defenders = [seat for seat in range(3) if seat != soloist]
        capacities = {defenders[0]: 10, defenders[1]: 10, SKAT_SLOT: 2}
        worlds = []
        for _ in range(count):
            deal = deal_unseen(unseen, capacities, {}, self.rng)
            hands: list[tuple[Card, ...]] = [(), (), ()]
            for defender in defenders:
                hands[defender] = tuple(deal[defender])
            skat = tuple(deal[SKAT_SLOT])
            assert len(skat) == 2
            worlds.append(_World(tuple(hands), skat))  # type: ignore[arg-type]
        return worlds

    def _estimate(
        self,
        cards: set[Card],
        soloist: int,
        dealer: int,
        game: Game,
        worlds: list[_World],
        simulations: int,
    ) -> float:
        per_world = max(1, simulations // len(worlds))
        probabilities = [
            self._world_probability(cards, soloist, dealer, game, world, per_world)
            for world in worlds
        ]
        return sum(probabilities) / len(probabilities)

    def _world_probability(
        self,
        cards: set[Card],
        soloist: int,
        dealer: int,
        game: Game,
        world: _World,
        iterations: int,
    ) -> float:
        twelve = sorted([*cards, *world.skat], key=lambda card: card.np_index)
        pressed = press_for_game(twelve, game)
        solo_hand = sorted(
            (card for card in twelve if card not in pressed),
            key=lambda card: card.np_index,
        )
        hands = [sorted(hand, key=lambda card: card.np_index) for hand in world.hands]
        hands[soloist] = solo_hand
        state = PlayoutState(
            trick=copy_trick(game.trick, keep_buffer=False),
            hands=hands,
            to_move=(dealer + 1) % 3,
            points=[
                sum(card.value for card in pressed) if seat == soloist else 0
                for seat in range(3)
            ],
            soloist=soloist,
        )
        root_player = state.to_move
        tally = uct_search(
            state,
            iterations,
            self.rng,
            self.exploration,
            outcome_only=True,
        )
        root_probability = max(
            value_sum / visits for visits, value_sum in tally.values() if visits
        )
        return root_probability if root_player == soloist else 1.0 - root_probability
