from __future__ import annotations

import os
from typing import Optional

import torch

from skat.agents.belief.checkpoint import load_belief
from skat.agents.belief.loss import belief_probabilities
from skat.agents.belief_recorder import belief_input
from skat.agents.bidding import BiddingStyle
from skat.agents.mcts import MCTSAgent
from skat.agents.rl.dqn import describe_device, resolve_device
from skat.agents.search import SKAT_SLOT, deal_unseen
from skat.card import Card
from skat.hand import FULL_HAND
from skat.models.belief import NEXT, PREVIOUS, SKAT

DEFAULT_BELIEF_PATH = "./trained_models/belief.pt"

COLLECT_HINT = (
    "  python -m skat --rounds 20000 -p0 policy -p1 policy -p2 policy "
    "--policy trained_models/policy-transformer.pt --record-belief data/belief\n"
    "  python -m skat.agents.belief.train --data data/belief"
)


class BeliefMCTSAgent(MCTSAgent):
    """
    `MCTSAgent` with its determinizations drawn from a learned belief instead
    of uniformly.

    The base agent samples every world consistent with what it has seen with
    equal probability. Consistency is a weak filter: it rules out cards the
    observer holds, cards already played and suits a seat has shown void in,
    and nothing else. So a Grand declarer is as likely to be dealt no jacks as
    three, and the simulations spent on the worlds nobody would reach are
    simply lost.

    `skat.agents.belief.BeliefNet` estimates where each unseen card is, and
    that estimate becomes the sampling weight in `deal_unseen`. Everything else
    - the UCT trees, the aggregation across worlds, `best` scoring by mean
    value, the bidding and the skat handling - is inherited unchanged, which is
    the point: whatever this agent gains or loses against `MCTSAgent` is
    attributable to the sampling and to nothing else.

    The network is evaluated **once per decision**, not once per world.
    `determinize` runs `determinizations` times per card and the belief does
    not change between those calls, so the forward pass is hoisted into
    `tally`. At 20 worlds a move that is the difference between one inference
    and twenty.
    """

    def __init__(
        self,
        path: str = DEFAULT_BELIEF_PATH,
        device: str = "auto",
        simulations: int = 600,
        determinizations: int = 20,
        exploration: float = 1.4,
        seed=None,
        dropout: float = 0.1,
        bidding: BiddingStyle = BiddingStyle.CONSERVATIVE,
    ) -> None:
        super().__init__(
            simulations=simulations,
            determinizations=determinizations,
            exploration=exploration,
            seed=seed,
            bidding=bidding,
        )
        self.device = resolve_device(device)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} does not exist. A BeliefMCTSAgent without a belief is "
                "just an MCTSAgent with an extra import - use `-p0 mcts` if "
                "that is what you want. Otherwise collect and train:\n"
                f"{COLLECT_HINT} --out {path}"
            )
        self.net, self.meta = load_belief(path, self.device, dropout=dropout)
        self.net.eval()
        self._weights: Optional[dict[Card, dict[int, float]]] = None
        print(f"BeliefMCTSAgent: {path} on {describe_device(self.device)}")

    # ----------------------------------------------------------------- belief

    def beliefs(self) -> dict[Card, dict[int, float]]:
        """
        `{card: {slot: probability}}` for this position, in the slot numbering
        `deal_unseen` uses - absolute seat ids, plus `SKAT_SLOT`.

        The network answers egocentrically (next, previous, skat) because that
        is how the observation is built; the search deals into absolute seats.
        This is the one place the two numberings meet.
        """
        assert self.state is not None and self.state.public_state is not None
        seat = self.state.seat_id
        # Built by the same helper the recorder used, voids and all, so the
        # network is asked the question it was trained on.
        state = torch.tensor(
            belief_input(self.state.public_state, seat), dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            probabilities = belief_probabilities(self.net(state.to(self.device)))[0]
        probabilities = probabilities.cpu()
        slot_of = {
            NEXT: (seat + 1) % 3,
            PREVIOUS: (seat + 2) % 3,
            SKAT: SKAT_SLOT,
        }
        return {
            card: {
                slot_of[location]: float(probabilities[card.np_index][location])
                for location in (NEXT, PREVIOUS, SKAT)
            }
            for card in FULL_HAND
        }

    # ------------------------------------------------------------------- play

    def tally(self, moves: list[Card]) -> dict[Card, list[float]]:
        # One forward pass for the whole decision, then `determinizations`
        # worlds drawn from it.
        self._weights = self.beliefs()
        try:
            return super().tally(moves)
        finally:
            self._weights = None

    def deal(self, unseen, capacities, voids) -> dict[int, list[Card]]:
        """Sample a world from the belief rather than uniformly."""
        return deal_unseen(unseen, capacities, voids, self.rng, weights=self._weights)
