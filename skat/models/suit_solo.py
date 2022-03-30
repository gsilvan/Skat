import torch.nn as nn


class SuitSoloNet(nn.Module):
    __hand = 32  # 32 cards
    __color = 5  # clubs, spades, hearts, diamonds
    __points = 2  # soloist, defender 1 + defender 2
    # __free_colors = 15  # solist, defender 1, defender 2
    __trick_value = 1
    __played_cards = 96

    INPUT_SIZE = __hand + __color + __points + __trick_value + __played_cards
    OUTPUT_SIZE = 32

    def __init__(
        self,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        dropout=0.25,
    ) -> None:
        super().__init__()
        self.linear_dropout_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=input_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(input_size, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        x = self.linear_dropout_relu_stack(x)
        return x
