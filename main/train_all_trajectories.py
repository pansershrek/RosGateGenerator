#!/usr/bin/env python3
import os

import torch

from train_one_trajectory import train_one_trajectory
from val_all_trajectories import val_all_trajectories

def train_all_trajectories(
    model, train_dataloader, val_dataloader, optimizer, criterion_coord,
    criterion_contacs, device, writer, epochs, scheduler, model_checkpoints
):
    for epoch in range(epochs):
        losses = []
        print(f"Start epoch: {epoch}")
        for trajectory in train_dataloader:
            loss = train_one_trajectory(
                model, trajectory, optimizer, criterion_coord,
                criterion_contacs, device, writer, epoch, scheduler
            )
            losses.append(loss)
        if writer is not None:
            writer.add_scalar(
                f"train/loss",
                sum(losses)/len(losses), epoch
            )
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
        )
        val_all_trajectories(
            model, val_dataloader, criterion_coord,
            criterion_contacs, device, writer, epoch
        )
