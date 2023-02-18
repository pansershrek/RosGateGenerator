#!/usr/bin/env python3

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor

def train_one_trajectory(
    model, trajectory, optimizer_coord, optimizer_contacs,
    device, writer, epochs, scheduler
)-> None:
    model.train()
    shift_tensor = create_shift_tesor(trajectory["shift"])
    h, c = None, None
    losses = []
    for point_idx in len(trajectory["points"]) - 1:
        coords_tensor_cur, contact_tensor_cur = (
            create_tensor_from_trajectory_point(
                trajectory["points"][point_idx]
            )
        )
        coords_tensor_next, contact_tensor_next = (
            create_tensor_from_trajectory_point(
                trajectory["points"][point_idx + 1]
            )
        )
        input_tensor = torch.cat((shift_tensor, coords_tensor_cur, contact_tensor_cur))

        optimizer.zero_grad()

        coords_tensor_pred, contact_tensor_pred, h, c = model(
            input_tensor.to(device), h, c
        )

        loss_coord = optimizer_coord(coords_tensor_pred, coords_tensor_next)
        loss_contacs = optimizer_contacs(contact_tensor_pred, contact_tensor_next)
        loss = loss_coord + loss_contacs

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if writer is not None:
            writer.add_scalar(
                f"train_one_trajectory/loss/step/trajectory_{trajectory['trajectory_idx']}",
                loss, point_idx
            )
    if writer is not None:
        writer.add_scalar(
            f"train_one_trajectory/loss/mean/trajectory_{trajectory['trajectory_idx']}",
            sum(losses)/len(losses), epochs
        )