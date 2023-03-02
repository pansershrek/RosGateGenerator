#!/usr/bin/env python3
import os

import torch

from train_one_trajectory import train_one_trajectory
from val import val

def train(
    model, train_dataloader, val_dataloader, optimizer, criterion_coord,
    device, writer, epochs, scheduler, model_checkpoints
):
    model.train()
    for epoch in range(epochs):
        losses = []
        print(f"Start epoch: {epoch}")
        for step, trajectory in enumerate(train_dataloader):
            h, c = None, None
            for trajectory_step_idx in range(1, trajectory["points"].shape[1]):
                masks = (trajectory["masks"][:, trajectory_step_idx] == 1.0).view(-1)
                if int(sum(masks)) == 0:
                    break

                optimizer.zero_grad()

                predict_points, h, c = model(
                    trajectory["points"][:, trajectory_step_idx].to(device),
                    h, c
                ) # Maybe use `:trajectory_step_idx` instead `trajectory_step_idx` to train on all history

                points_idx = (
                    trajectory["points"].shape[2] -
                    trajectory["shift"].shape[1]
                ) // 2
                loss = criterion_coord(
                    predict_points[masks],
                    trajectory["points"][
                        masks, trajectory_step_idx, - points_idx :
                    ].to(device)
                )

                h = h.detach()
                c = c.detach()

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if scheduler is not None:
                    scheduler.step()

                if writer is not None:
                    writer.add_scalar(
                        "train/cur_loss",
                        loss.item(),
                        (
                            epoch * trajectory["points"].shape[1] +
                            trajectory_step_idx
                        )
                    )
        val(
            model, val_dataloader, criterion_coord,
            device, writer, epoch
        )
        if writer is not None:
            writer.add_scalar(
                "train/loss",
                sum(losses)/len(losses), epoch
            )
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
        )
