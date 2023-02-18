#!/usr/bin/env python3
import os

import torch

from train_one_trajectory import train_one_trajectory
from val import val

def train(
    model, train_dataloader, val_dataloader, criterion, optimizer_coord,
    optimizer_contacs, device, writer, epochs, scheduler, model_checkpoints
):
    for trajectory in train_dataloader:
        train_one_trajectory(
            model, trajectory, criterion, optimizer_coord,
            optimizer_contacs, device, writer, epochs, scheduler
        )
        break
    torch.save(
        model.state_dict(),
        os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
    )
    #val()
