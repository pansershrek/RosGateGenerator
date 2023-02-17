#!/usr/bin/env python3
from train_one_trajectory import train_one_trajectory

def train(
    model, train_dataloader, val_dataloader, criterion, optimizer, device,
    writer, epochs, scheduler
):
    for trajectory in train_dataloader:
        train_one_trajectory(
            model, trajectory, criterion, optimizer,
            device, writer, epochs, scheduler
        )
        return None