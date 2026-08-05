from torch import nn

from skat.models.suit_solo import SuitSoloNet


class PolicyNet(nn.Module):
    """
    Card-play policy: the 240 observation floats in, 32 logits out - one per
    card slot, to be read under a legal-move mask.

    Same geometry as `SuitSoloNet` and deliberately a separate class. Its
    outputs are policy logits, not Q-values: `SuitSoloNet` is regressed against
    `Round.cumulative_reward`, this is trained against MCTS visit counts, and
    the two mean different things. Sharing the class would make the checkpoints
    interchangeable at `load_state_dict` and silently wrong at play time -
    `skat.agents.distill.checkpoint` is the other half of stopping that.

    Two differences from `SuitSoloNet`, both on purpose:

    - Fixed `nn.Linear` throughout rather than `nn.LazyLinear`. Lazy layers
      materialize from whatever state dict they are handed, so a checkpoint
      from a different state layout loads cleanly and fails much later; these
      fail at load.
    - Dropout that exists. `SuitSoloNet` takes a `dropout` argument, names its
      stack after it and contains no `Dropout` layer. Three million parameters
      against ~100k samples, of which each group of ~22 comes from one round
      and is highly correlated, will overfit without one.
    """

    INPUT_SIZE = SuitSoloNet.INPUT_SIZE
    OUTPUT_SIZE = SuitSoloNet.OUTPUT_SIZE

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
