#!/usr/bin/env python3
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(
        self, input_size: int = 2*(95 + 4) + 3,
        hidden_size: int = 256, num_layers: int = 3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers
        )
        self.fc1 = nn.Linear(hidden_size, 95 + 4) # shape of coord_tensor + contact_tensor
        self.rl = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, c):
        if h is not None and c is not None:
            output, (h, c) = self.lstm(
                x.view(1, -1), (h, c)
            )
        else:
            output, (h, c) = self.lstm(x.view(1, -1))
        output = self.fc1(self.rl(output)).view(-1)
        coord_tensor, contacs_tensor = output[:95], output[95:]
        return coord_tensor, self.sigmoid(contacs_tensor), h, c