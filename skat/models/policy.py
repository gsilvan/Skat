from torch import nn

from skat.models.observation import INPUT_SIZE, OUTPUT_SIZE


class PolicyNet(nn.Module):
    """
    Card-play policy: the 240 observation floats in, 32 logits out - one per
    card slot, to be read under a legal-move mask.

    Same geometry as `QNet` and deliberately a separate class. Its
    outputs are policy logits, not Q-values: `QNet` is regressed against
    `Round.cumulative_reward`, this is trained against MCTS visit counts, and
    the two mean different things. Sharing the class would make the checkpoints
    interchangeable at `load_state_dict` and silently wrong at play time -
    `skat.agents.distill.checkpoint` is the other half of stopping that.

    The one structural difference is dropout, and it goes the way it does
    because the two are trained differently. `QNet` has none on purpose:
    dropout inside a Q-network perturbs the bootstrap target as well as the
    prediction, which is a poor trade in TD learning. This net is doing
    supervised classification against a fixed corpus, where three million
    parameters against ~100k samples - each group of ~22 coming from one round
    and highly correlated - will overfit without it.
    """

    INPUT_SIZE = INPUT_SIZE
    OUTPUT_SIZE = OUTPUT_SIZE

    WIDTHS = (1024, 1024, 1024, 512, 128)

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        output_size: int = OUTPUT_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for width in self.WIDTHS:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, output_size))
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)
