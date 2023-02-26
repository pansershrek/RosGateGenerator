#!/usr/bin/env python3

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor

def val(
    model, val_dataloader, criterion_coord,
    device, writer, epoch
):
    model.eval()
    losses = []
    losses_final_point = []
    h, c = None, None

    for step, trajectory in enumerate(val_dataloader):
        predict_final_point = []
        real_final_point = []
        for trajectory_step_idx in range(1, trajectory["points"].shape[1]):
            masks = (trajectory["masks"][:, trajectory_step_idx] == 1.0).view(-1)
            if int(sum(masks)) == 0:
                break

            predict_points, h, c = model(
                trajectory["points"][:, trajectory_step_idx].to(device),
                h, c
            ) # Maybe use `:trajectory_step_idx` instead `trajectory_step_idx` to train on all history

            points_idx = (
                trajectory["points"].shape[2] -
                trajectory["shift"].shape[1]
            ) // 2
            loss = criterion_coord(
                predict_points.view(
                    [predict_points.shape[0], predict_points.shape[2]]
                )[masks],
                trajectory["points"][
                    masks, trajectory_step_idx, - points_idx :
                ].to(device)
            )
            # Save final points of trajectories
            if (
                trajectory_step_idx + 1 == trajectory["points"].shape[1]
            ):
                for masks_idx in range(len(masks)):
                    if masks[masks_idx] == True:
                        predict_final_point.append(
                            predict_points.view(
                                [
                                    predict_points.shape[0],
                                    predict_points.shape[2]
                                ]
                            )[masks_idx]
                        )
                        real_final_point.append(
                            trajectory["points"][
                                masks_idx, trajectory_step_idx, - points_idx :
                            ].to(device)
                        )

            if (
                trajectory_step_idx + 1 < trajectory["points"].shape[1]
            ):
                masks_next = (
                    trajectory["masks"][:, trajectory_step_idx + 1] == 1.0
                ).view(-1)
                for masks_idx in range(len(masks)):
                    if (
                        masks[masks_idx] == True and
                        masks_next[masks_idx] == False
                    ):
                        predict_final_point.append(
                            predict_points.view(
                                [
                                    predict_points.shape[0],
                                    predict_points.shape[2]
                                ]
                            )[masks_idx]
                        )
                        real_final_point.append(
                            trajectory["points"][
                                masks_idx, trajectory_step_idx, - points_idx :
                            ].to(device)
                        )


            h = h.detach()
            c = c.detach()

            losses.append(loss.item())

        losses_final_point.append(
            criterion_coord(
                torch.stack(predict_final_point, dim=0),
                torch.stack(real_final_point, dim=0)
            )
        )

    if writer is not None:
        writer.add_scalar(
            "val/loss",
            sum(losses) / len(losses), epoch
        )
        writer.add_scalar(
            "val/final_point_dist",
            sum(losses_final_point) / len(losses_final_point),
            epoch
        )

