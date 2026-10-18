import torch

from skat.agents import Agent
from skat.agents.distill.checkpoint import load_policy
from skat.hand import FULL_HAND
from skat.utils.reproducibility import resolve_device


class PolicyAgent(Agent):
    def __init__(self, path, device="cpu", seed=None):
        super().__init__(seed)
        self.device = resolve_device(device)
        self.net, self.metadata = load_policy(path, self.device)

    def choose_card(self, valid_moves):
        moves = sorted(valid_moves, key=lambda c: c.np_index)
        if len(moves) == 1:
            return moves[0]
        state = torch.tensor(self.state.observation, device=self.device).unsqueeze(0)
        mask = torch.zeros(32, dtype=torch.bool, device=self.device)
        mask[[c.np_index for c in moves]] = True
        with torch.inference_mode():
            logits = self.net(state)[0].masked_fill(~mask, -torch.inf)
        return FULL_HAND[int(logits.argmax())]
