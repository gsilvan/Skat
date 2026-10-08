from __future__ import annotations

import math
from random import Random
from typing import Optional

from skat.agents.search.playout import PlayoutState
from skat.card import Card

#: Textbook UCT constant for rewards scaled to [0, 1].
DEFAULT_EXPLORATION = 1.4


class Node:
    """
    One node of the UCT tree. `value_sum` is accumulated from the point of view
    of `moved_by` — the seat that chose the move leading here — so a parent
    picking among its children always reads values from its own perspective.
    """

    __slots__ = (
        "move",
        "moved_by",
        "to_move",
        "parent",
        "children",
        "untried",
        "visits",
        "value_sum",
    )

    def __init__(
        self,
        to_move: int,
        untried: list[Card],
        move: Optional[Card] = None,
        moved_by: Optional[int] = None,
        parent: Optional[Node] = None,
    ) -> None:
        self.move = move
        self.moved_by = moved_by
        self.to_move = to_move
        self.parent = parent
        self.children: list[Node] = []
        self.untried = untried
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def add_child(self, move: Card, state: PlayoutState) -> Node:
        child = Node(
            to_move=state.to_move,
            untried=state.legal_moves(),
            move=move,
            moved_by=self.to_move,
            parent=self,
        )
        self.children.append(child)
        return child

    def select_child(self, exploration: float) -> Node:
        log_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda child: (
                child.value + exploration * math.sqrt(log_visits / child.visits)
            ),
        )


def uct_search(
    state: PlayoutState,
    iterations: int,
    rng: Random,
    exploration: float = DEFAULT_EXPLORATION,
    outcome_only: bool = False,
) -> dict[Card, tuple[int, float]]:
    """
    Run `iterations` UCT playouts on the perfect-information `state` and report
    `{move: (visits, value_sum)}` for the root's moves, valued from the
    perspective of the seat to move at the root. `outcome_only` replaces the
    normal win-first/secured-bonus reward with strict party win/loss; bidding uses it to
    estimate a probability rather than a card-point utility.
    """
    root = Node(to_move=state.to_move, untried=state.legal_moves())
    for _ in range(iterations):
        node = root
        sim = state.clone()

        # selection: descend fully expanded nodes by UCB1
        while not node.untried and node.children:
            node = node.select_child(exploration)
            assert node.move is not None
            sim.apply(node.move)

        # expansion: try one unplayed move
        if node.untried:
            move = node.untried.pop(rng.randrange(len(node.untried)))
            sim.apply(move)
            node = node.add_child(move, sim)

        # simulation: uniformly random legal play to the end of the round
        while not sim.is_terminal:
            moves = sim.legal_moves()
            sim.apply(moves[rng.randrange(len(moves))])

        # backpropagation
        rewards = sim.outcome_rewards() if outcome_only else sim.rewards()
        current: Optional[Node] = node
        while current is not None:
            current.visits += 1
            if current.moved_by is not None:
                current.value_sum += rewards[current.moved_by]
            current = current.parent

    return {child.move: (child.visits, child.value_sum) for child in root.children}  # type: ignore[misc]
