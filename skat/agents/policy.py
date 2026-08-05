from __future__ import annotations

import os

import torch

from skat.agents import Agent
from skat.agents.bidding import BiddingStyle
from skat.agents.declaring import HeuristicDeclarer
from skat.agents.distill.checkpoint import load_policy
from skat.agents.distill.loss import masked_log_softmax
from skat.agents.rl.dqn import describe_device, resolve_device
from skat.card import Card
from skat.hand import FULL_HAND, Hand

DEFAULT_POLICY_PATH = "./trained_models/policy.pt"

COLLECT_HINT = (
    "  python -m skat --rounds 4500 -p0 mcts -p1 mcts -p2 mcts "
    "--record-mcts data/distill\n"
    "  python -m skat.agents.distill.train --data data/distill"
)


class PolicyAgent(HeuristicDeclarer, Agent):
    """
    Plays the card a policy network - distilled from `MCTSAgent` - likes best.

    One forward pass per decision instead of `simulations` playouts, which is
    two to three orders of magnitude cheaper than the teacher it was trained
    on. Like `DQNAgent` it learns card play only: bidding comes from
    `Agent`/`BiddingStyle` and the skat phase from `HeuristicDeclarer`.

    One checkpoint serves all three seats. `Round.get_state` is egocentric, so
    the same net is the right net in every chair - which is also why the
    corpus pools every seat into one file. That is what `--policy` means, as
    against the DQN's per-seat `-m{i}`.
    """

    def __init__(
        self,
        path: str = DEFAULT_POLICY_PATH,
        device: str = "auto",
        temperature: float = 0.0,
        seed=None,
        dropout: float = 0.1,
        bidding: BiddingStyle = BiddingStyle.CONSERVATIVE,
    ) -> None:
        super().__init__(bidding=bidding)
        self.device = resolve_device(device)
        if not os.path.exists(path):
            # `DQN` starts from a random init when its checkpoint is missing,
            # because it is about to train one. This agent never learns, so a
            # random init would just be a bad player that reads like a trained
            # one in `Tournament.summary()`. Refuse instead.
            raise FileNotFoundError(
                f"{path} does not exist, and a PolicyAgent has nothing to fall "
                "back on - it never trains. Collect and distil first:\n"
                f"{COLLECT_HINT} --out {path}"
            )
        self.net, self.meta = load_policy(path, self.device, dropout=dropout)
        self.net.eval()
        self.temperature = temperature
        self.rng: torch.Generator | None = None
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        print(f"PolicyAgent: {path} on {describe_device(self.device)}")

    def choose_card(self, valid_moves: set[Card]) -> Card:
        if self.state is None:
            raise Exception("Can't choose a card without having a state.")
        choice = self.select(valid_moves)
        self.state.hand.remove(choice)
        return choice

    def select(self, valid_moves: set[Card]) -> Card:
        """Pick a card from `valid_moves` without mutating the hand."""
        moves = list(valid_moves)
        if len(moves) == 1:
            # Forced. The teacher never searched these and the corpus holds
            # none of them, so there is nothing to ask the network about.
            return moves[0]
        assert self.state is not None and self.state.public_state is not None
        state = self.state.public_state.get_state_t(player_id=self.state.seat_id)
        mask = Hand(tuple(moves)).as_tensor_mask.unsqueeze(0)
        with torch.no_grad():
            logits = self.net(state.to(self.device))
        log_p = masked_log_softmax(logits, mask.to(self.device))
        if self.temperature > 0:
            probs = torch.softmax(log_p / self.temperature, dim=-1)
            index = int(torch.multinomial(probs[0].cpu(), 1, generator=self.rng))
        else:
            index = int(log_p.argmax())
        return FULL_HAND[index]
