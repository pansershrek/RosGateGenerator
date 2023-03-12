#!/usr/bin/env python3
import os
import logging

import torch

from main_utils import quaternion_to_euler_torch, get_pred_base_points_and_angle
from val import val

def train(
    model, train_dataloader, val_dataloader, optimizer, criterion_coord,
    criterion_angle, device, writer, epochs, scheduler, model_checkpoints
):
    for epoch in range(epochs):
        losses = []
        losses_base_point = []
        losses_base_angle = []
        logging.warning(f"Train epoch: {epoch}")
        model.train()
        for step, trajectory in enumerate(train_dataloader):
            h, c = None, None
            for trajectory_step_idx in range(1, trajectory["points"].shape[1]):
                masks = (trajectory["masks"][:, trajectory_step_idx] == 1.0).view(-1)
                if int(sum(masks)) == 0:
                    break

                optimizer.zero_grad()

                predict_points, _, _ = model(
                    trajectory["points"][:, :trajectory_step_idx].to(device),
                    h, c
                ) # Maybe use `:trajectory_step_idx` instead `trajectory_step_idx` to train on all history
                predict_points = torch.squeeze(predict_points)

                points_idx = (
                    trajectory["points"].shape[2] -
                    trajectory["shift"].shape[1]
                ) // 2
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
                #h = h.detach()
                #c = c.detach()

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
            writer.add_scalar(
                "train/losses_base_point",
                sum(losses_base_point)/len(losses_base_point), epoch
            )
            writer.add_scalar(
                "train/losses_base_angle",
                sum(losses_base_angle)/len(losses_base_angle), epoch
            )
        val(
            model, val_dataloader, criterion_coord, criterion_angle,
            device, writer, epoch
        )
        logging.warning(f"Train loss is: {sum(losses)/len(losses)}")
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
        )
