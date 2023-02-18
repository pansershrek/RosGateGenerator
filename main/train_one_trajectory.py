#!/usr/bin/env python3

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor

def train_one_trajectory(
    model, trajectory, optimizer, criterion_coord, criterion_contacs,
    device, writer, epoch, scheduler
):
    model.train()
    shift_tensor = create_shift_tesor(trajectory["shift"])
    coords_tensor_start, contact_tensor_start = (
        create_tensor_from_trajectory_point(
            trajectory["points"][0]
        )
    )
    #coords_tensor_end, contact_tensor_end = (
    #    create_tensor_from_trajectory_point(
    #        trajectory["points"][0]
    #    )
    #)
    state_tensor = torch.cat((shift_tensor, coords_tensor_start, contact_tensor_start))
    h, c = None, None
    losses = []
    for point_idx in range(len(trajectory["points"]) - 1):
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
        input_tensor = torch.cat((state_tensor, coords_tensor_cur, contact_tensor_cur))

        optimizer.zero_grad()

        coords_tensor_pred, contact_tensor_pred, h, c = model(
            input_tensor.to(device), h, c
        )

        loss_coords = criterion_coord(
            coords_tensor_pred, coords_tensor_next.to(device)
        )
        loss_contacs = criterion_contacs(
            contact_tensor_pred, contact_tensor_next.to(device)
        )
        loss = loss_coords + loss_contacs
        loss.backward() # Remove detach and add loss.backward(retain_graph=True)
        h = h.detach()
        c = c.detach()
        optimizer.step()
        losses.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        if writer is not None:
            writer.add_scalar(
                f"train_one_trajectory_step_loss/epoch_{epoch}/trajectory_{trajectory['trajectory_idx']}",
                loss, point_idx
            )
    if writer is not None:
        writer.add_scalar(
            f"train_one_trajectory_mean_loss/trajectory_{trajectory['trajectory_idx']}",
            sum(losses)/len(losses), epoch
        )
    return sum(losses) / len(losses)