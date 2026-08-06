from torch import nn

from skat.models.observation import INPUT_SIZE, OUTPUT_SIZE


class QNet(nn.Module):
    """
    The DQN's action-value network: the observation in, one Q-value per card
    out, regressed against `Round.cumulative_reward`.

    Sits alongside `PolicyNet` and `BeliefNet`, which are the same shape and
    mean entirely different things - policy logits and card locations
    respectively. That is not a naming nicety: all three map 240 floats to 32
    numbers, so nothing about the tensors distinguishes them and a checkpoint
    loaded into the wrong one plays confidently and wrongly. `PolicyNet`'s
    checkpoints are wrapped in a format marker for exactly that reason; this
    one is saved as a bare state dict by `skat.agents.rl.dqn.DQN`, which is
    what the marker distinguishes it from.

    It was called `SuitSoloNet`, from a time when it only played suit games.
    It has handled grand and null for as long as the game block has carried
    `is_grand` and `is_null` flags, so the name described neither its input
    nor its output.

    No dropout, on purpose. The constructor used to take a `dropout` argument
    and name the stack after it while containing no `Dropout` layer at all -
    and the right resolution is to drop the argument rather than wire it up.
    Dropout inside a Q-network perturbs the bootstrap target as well as the
    prediction, which is a poor trade in TD learning. `PolicyNet` has real
    dropout because it is doing supervised classification against a fixed
    corpus, where the trade goes the other way.
    """

    INPUT_SIZE = INPUT_SIZE
    OUTPUT_SIZE = OUTPUT_SIZE

    WIDTHS = (1024, 1024, 1024, 512, 128)

    def __init__(
        self, input_size: int = INPUT_SIZE, output_size: int = OUTPUT_SIZE
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for width in self.WIDTHS:
            # Fixed `Linear`, not `LazyLinear`. A lazy layer materialises from
            # whatever state dict it is handed, so a checkpoint from a
            # different observation layout loads cleanly and goes wrong later;
            # this one fails at load, which is where it should.
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, output_size))
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)
