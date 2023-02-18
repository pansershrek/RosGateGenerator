#!/usr/bin/env python3
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(
        self, input_size: int = 95 + 4 + 3,
        hidden_size: int = 256, num_layers: int = 3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers
        )

    def forward(self, x, h, c):
        if h is not None and c is not None:
            output, (h, c) = self.lstm(x, (h, c))
        else:
            output, (h, c) = self.lstm(x, (h, c))
        coord_tensor, leg_contacs = torch.split(output)
        return coord_tensor, leg_contacs, h, c