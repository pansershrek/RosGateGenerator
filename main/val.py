#!/usr/bin/env python3

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor

def val(
    model, val_dataloader, criterion_coord,
    criterion_contacs, device, writer, epoch
):
    model.eval()
    coords_losses_mean = []
    contact_losses_mean = []
    contact_acc_mean = []
    for trajectory in val_dataloader:
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
        coords_loss = 0
        contact_loss = 0
        contact_acc = 0
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

            coords_tensor_pred, contact_tensor_pred, h, c = model(
                input_tensor.to(device), h, c
            )

            loss_coords = criterion_coord(
                coords_tensor_pred, coords_tensor_next.to(device)
            )
            loss_contacs = criterion_contacs(
                contact_tensor_pred, contact_tensor_next.to(device)
            )
            coords_loss += loss_coords.item()
            contact_loss += loss_contacs.item()
            contact_acc += int(sum(contact_tensor_next == (contact_tensor_pred >= 0.5)))

            h = h.detach()
            c = c.detach()
        coords_losses_mean.append(coords_loss)
        contact_losses_mean.append(contact_loss)
        contact_acc_mean.append(contact_acc / (len(trajectory["points"]) - 1))

    if writer is not None:
        writer.add_scalar(
            f"val/coords_losses_mean",
            sum(coords_losses_mean)/len(coords_losses_mean), epoch
        )
        writer.add_scalar(
            f"val/contact_losses_mean",
            sum(contact_losses_mean)/len(contact_losses_mean), epoch
        )
        writer.add_scalar(
            f"val/contact_acc_mean",
            sum(contact_acc_mean)/len(contact_acc_mean), epoch
        )
