#!/usr/bin/env python3

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor

def val(
    model, val_dataloader, criterion_coord,
    device, writer, epoch
):
    model.eval()
    losses = []
    h, c = None, None

    for step, trajectory in enumerate(val_dataloader):
        for trajectory_step_idx in range(1, trajectory["points"].shape[1]):
            predict_points, h, c = model(
                trajectory["points"][:, :trajectory_step_idx].to(device),
                h, c
            )

            points_idx = (
                trajectory["points"].shape[2] -
                trajectory["shift"].shape[1]
            ) // 2
            loss = criterion_coord(
                predict_points.view(
                    [predict_points.shape[0], predict_points.shape[2]]
                ),
                trajectory["points"][
                    :, trajectory_step_idx, - points_idx :
                ].to(device)
            )

            h = h.detach()
            c = c.detach()

            losses.append(loss.item())

    if writer is not None:
        writer.add_scalar(
            f"val/loss",
            sum(losses)/len(losses), epoch
        )

