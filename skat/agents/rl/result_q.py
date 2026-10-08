"""Independent Q-functions for winning and bonuses after secured victory."""

from torch import nn


class ResultQ(nn.Module):
    def __init__(self, architecture, kwargs):
        super().__init__()
        # No shared weights: fitting bonuses cannot perturb win predictions.
        self.win = architecture(**kwargs)
        self.bonus = architecture(**kwargs)

    def forward(self, state):
        return self.win(state)


def secured_victory(states):
    """Read role/party points from egocentric observations; exclude Null.

    Overbids are decided before card play; DQNAgent explicitly disables bonus
    selection there because the observation does not encode the bid.
    """
    solo = states[:, 38] > 0.5
    points = states[:, 42] * 120
    return (states[:, 37] < 0.5) & (
        (solo & (points > 60.5)) | (~solo & (points > 59.5))
    )


def resolved_result(states):
    """Once either party has secured victory, its primary return is settled."""
    solo = states[:, 38] > 0.5
    theirs = states[:, 43] * 120
    other_won = (states[:, 37] < 0.5) & (
        (solo & (theirs > 59.5)) | (~solo & (theirs > 60.5))
    )
    return secured_victory(states) | other_won
