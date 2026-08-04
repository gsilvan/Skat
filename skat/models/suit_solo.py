import torch.nn as nn


class SuitSoloNet(nn.Module):
    # Listed in the order `Round.get_state` concatenates them - the two are
    # coupled by hand, so keeping them in step is the only thing that keeps the
    # coupling readable. `tests/test_state_vector.py` asserts the total matches.
    __hand = 32  # own cards
    __game = 6  # trump one-hot, is_grand, is_null
    __role = 1  # am I the soloist
    __soloist_seat = 3  # where the soloist sits, relative to me
    __points = 2  # my party, their party
    __played_cards = 96  # 3 x 32, slot 0 = me
    __current_trick = 96  # 3 x 32, slot 0 = me
    __front_hand = 3  # front / middle / back hand
    __trick_value = 1

    INPUT_SIZE = (
        __hand
        + __game
        + __role
        + __soloist_seat
        + __points
        + __played_cards
        + __current_trick
        + __front_hand
        + __trick_value
    )
    OUTPUT_SIZE = 32

    def __init__(
        self,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        dropout=0.333,
    ) -> None:
        super().__init__()
        self.linear_dropout_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(output_size),
        )

    def forward(self, x):
        x = self.linear_dropout_relu_stack(x)
        return x
