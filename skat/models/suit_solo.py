import torch.nn as nn
import torch.nn.functional as F

__hand = 32  # 32 cards
__color = 5  # clubs, spades, hearts, diamonds
__points = 2  # soloist, defender 1 + defender 2
__free_colors = 15  # solist, defender 1, defender 2
__trick_value = 1
__played_cards = 96

_inputs = __hand + __color + __points + __free_colors + __trick_value + __played_cards


class SuitSoloNet(nn.Module):
    def __init__(
        self,
        input_size=_inputs,
        output_size=32,
        hidden_activation=F.relu,
        output_activation=F.softmax,
        dropout=0.25,
    ) -> None:
        super().__init__()
        self.activation = hidden_activation
        self.output_activation = output_activation
        self.input = nn.Linear(in_features=input_size, out_features=output_size)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.drop_p = dropout

    def forward(self, x):
        x = self.input(x)
        x = self.fc1(x)
        x = F.dropout(x, self.drop_p)
        x = self.activation(x)
        x = self.fc2(x)
        x = F.dropout(x, self.drop_p)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.output_activation(x, -1)
        return x
