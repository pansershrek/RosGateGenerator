#!/usr/bin/env python3
import logging

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor
from main_utils import quaternion_to_euler_torch, get_pred_base_points_and_angle

def val_ff(
    model, val_dataloader, criterion_coord,
    criterion_angle, device, writer, epoch
):
    model.eval()
    losses = []
    losses_final_point = []
    losses_final_point_base = []
    losses_final_angle = []
    losses_final_angle_base = []
    losses_base_point = []
    losses_base_angle = []
    logging.warning(f"Val epoch: {epoch}")
    for step, trajectory in enumerate(val_dataloader):
        predict_final_point = []
        real_final_point = []
        predict_final_point_base = []
        real_final_point_base = []
        predict_final_angle_base = []
        real_final_angle_base = []
        h, c = None, None
        for trajectory_step_idx in range(1, trajectory["points"].shape[1]):
            logging.warning(f"Val step: {trajectory_step_idx}")
            masks = (trajectory["masks"][:, trajectory_step_idx] == 1.0).view(-1)
            if int(sum(masks)) == 0:
                break

            predict_points = model(
                trajectory["points"][:, trajectory_step_idx].to(device)
            ) # Maybe use `:trajectory_step_idx` instead `trajectory_step_idx` to train on all history
            predict_points = torch.squeeze(predict_points)

            points_idx = (
                trajectory["points"].shape[2] -
                trajectory["shift"].shape[1]
            ) // 2
            points_idx = 35
            target_point = trajectory["points"][
                :, trajectory_step_idx, - points_idx :
            ]

            loss_full = criterion_coord(
                predict_points[masks],
                target_point[masks].to(device)
            )

            (
                pred_base_points, pred_base_angle
            ) = get_pred_base_points_and_angle(predict_points, masks)
            (
                gt_base_points, gt_base_angle
            ) = get_pred_base_points_and_angle(target_point, masks)

            loss_base_point = criterion_coord(
                pred_base_points,
                gt_base_points.to(device)
            )
            loss_base_angle = criterion_angle(
                quaternion_to_euler_torch(pred_base_angle),
                quaternion_to_euler_torch(gt_base_angle).to(device)
            )

            losses_base_point.append(loss_base_point.item())
            losses_base_angle.append(loss_base_angle.item())
            loss = loss_full + loss_base_point + loss_base_angle

            # Save final points of trajectories
            if (
                trajectory_step_idx + 1 == trajectory["points"].shape[1]
            ):
                for masks_idx in range(len(masks)):
                    if masks[masks_idx] == True:
                        predict_final_point.append(
                            predict_points[masks_idx]
                        )
                        real_final_point.append(
                            target_point[masks_idx].to(device)
                        )

                        (
                            pred_base_points, pred_base_angle
                        ) = get_pred_base_points_and_angle(predict_points, masks_idx)
                        (
                            gt_base_points, gt_base_angle
                        ) = get_pred_base_points_and_angle(target_point, masks_idx)

                        predict_final_point_base.append(pred_base_points)
                        real_final_point_base.append(gt_base_points.to(device))

                        predict_final_angle_base.append(pred_base_angle)
                        real_final_angle_base.append(gt_base_angle.to(device))

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
                            predict_points[masks_idx]
                        )
                        real_final_point.append(
                            target_point[masks_idx].to(device)
                        )

                        (
                            pred_base_points, pred_base_angle
                        ) = get_pred_base_points_and_angle(predict_points, masks_idx)
                        (
                            gt_base_points, gt_base_angle
                        ) = get_pred_base_points_and_angle(target_point, masks_idx)

                        predict_final_point_base.append(pred_base_points)
                        real_final_point_base.append(gt_base_points.to(device))

                        predict_final_angle_base.append(pred_base_angle)
                        real_final_angle_base.append(gt_base_angle.to(device))

            #h = h.detach()
            #c = c.detach()

            losses.append(loss.item())


        losses_final_point.append(
            criterion_coord(
                torch.stack(predict_final_point, dim=0),
                torch.stack(real_final_point, dim=0)
            )
        )
        losses_final_point_base.append(
            criterion_coord(
                torch.stack(predict_final_point_base, dim=0),
                torch.stack(real_final_point_base, dim=0)
            )
        )
        losses_final_angle_base.append(
            criterion_coord(
                torch.stack(predict_final_angle_base, dim=0),
                torch.stack(real_final_angle_base, dim=0)
            )
        )
    logging.warning(f"Val loss is: {sum(losses) / len(losses)}")
    logging.warning(f"Val final point dist is: {sum(losses_final_point) / len(losses_final_point)}")
    logging.warning(f"Val final point base dist is: {sum(losses_final_point_base) / len(losses_final_point_base)}")

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
        writer.add_scalar(
            "val/final_point_base_dist",
            sum(losses_final_point_base) / len(losses_final_point_base),
            epoch
        )
        writer.add_scalar(
            "val/final_point_angle_dist",
            sum(losses_final_angle_base) / len(losses_final_angle_base),
            epoch
        )
        writer.add_scalar(
            "train/losses_base_point",
            sum(losses_base_point)/len(losses_base_point), epoch
        )
        writer.add_scalar(
            "train/losses_base_angle",
            sum(losses_base_angle)/len(losses_base_angle), epoch
        )