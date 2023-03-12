#!/usr/bin/env python3
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(
        self, input_size: int = 2 * 35 + 3,
        output_size: int = 35, hidden_size: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x, h = None, c = None):
        for idx in range(x.shape[1]):
            if h is not None and c is not None:
                output, (h, c) = self.lstm(
                    x[:, idx], (h, c)
                )
            else:
                output, (h, c) = self.lstm(x[:, idx])
        output = self.fc1(self.relu(output))
        return output, h, c