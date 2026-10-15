"""One belief inference per decision, with exact constrained world sampling."""

import numpy as np
import torch

from skat.agents.belief.checkpoint import load_belief
from skat.agents.belief.features import belief_input
from skat.agents.search.samplers import PreparedWeightedSampler


class BeliefSampler:
    def __init__(self, path, device="cpu"):
        self.model, self.metadata = load_belief(path, device)
        self.device = device
        self.temperature = float(self.metadata.get("sampling_strength", 1.0))
        if not 0 <= self.temperature <= 2:
            raise ValueError("invalid belief sampling strength")

    def prepare(self, view):
        features = torch.from_numpy(belief_input(view)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(features)[0].cpu().numpy()
        logits = self.temperature * (logits - logits.max(axis=1, keepdims=True))
        weights = np.clip(np.exp(logits), 1e-4, 1.0)
        slots = ((view.seat_id + 1) % 3, (view.seat_id + 2) % 3)
        return PreparedWeightedSampler(
            {
                card: {
                    slots[0]: float(row[0]),
                    slots[1]: float(row[1]),
                    -1: float(row[2]),
                }
                for card, row in enumerate(weights)
            }
        )
