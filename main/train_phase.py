#!/usr/bin/env python3
import os
import logging

import torch

from main_utils import quaternion_to_euler_torch, get_pred_base_points_and_angle
from val_phase import val_phase

def train_phase(
    model, train_dataloader, val_dataloader, optimizer, criterion,
    device, writer, epochs, scheduler, model_checkpoints
):

    for epoch in range(epochs):
        losses = []
        logging.warning(f"Train epoch: {epoch}")
        model.train()
        for step, data in enumerate(train_dataloader):
            logging.warning(f"Train step: {step}")

            optimizer.zero_grad()
            output = model(
                data["features"].to(device),
            )

            loss = criterion(
                output,
                data["targets"].to(device)
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if writer is not None:
            writer.add_scalar(
                "train/loss",
                sum(losses)/len(losses), epoch
            )

        val_phase(
            model, val_dataloader, criterion,
            device, writer, epoch
        )
        logging.warning(f"Train loss is: {sum(losses)/len(losses)}")
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
        )
