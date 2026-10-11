"""Determinized UCT using binary party victories only."""

import math

DEFAULT_EXPLORATION = 1.4


class Node:
    def __init__(self, state, move=None, moved_by=None):
        self.move, self.moved_by = move, moved_by
        self.to_move = state.to_move
        self.untried = [] if state.is_terminal else state.legal_moves()
        self.children = []
        self.visits = 0
        self.value_sum = 0.0

    def select_child(self):
        return max(
            self.children,
            key=lambda c: (
                c.value_sum / c.visits
                + DEFAULT_EXPLORATION * math.sqrt(math.log(self.visits) / c.visits)
            ),
        )


def uct_search(state, iterations, rng):
    if iterations < 1 or state.is_terminal:
        return {}
    root = Node(state)
    for _ in range(iterations):
        node = root
        sim = state.clone()
        path = [node]
        while not node.untried and node.children:
            node = node.select_child()
            sim.apply(node.move)
            path.append(node)
        if node.untried:
            move = node.untried.pop(rng.randrange(len(node.untried)))
            moved_by = sim.to_move
            sim.apply(move)
            child = Node(sim, move, moved_by)
            node.children.append(child)
            node = child
            path.append(node)
        while not sim.is_terminal:
            sim.apply(rng.choice(sim.legal_moves()))
        rewards = sim.rewards()
        for node in path:
            node.visits += 1
            if node.moved_by is not None:
                node.value_sum += rewards[node.moved_by]
    return {c.move: (c.visits, c.value_sum) for c in root.children}
