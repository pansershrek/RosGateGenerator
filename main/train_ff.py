#!/usr/bin/env python3
import os
import logging

import torch

from main_utils import quaternion_to_euler_torch, get_pred_base_points_and_angle
from val_ff import val_ff

def train_ff(
    model, train_dataloader, val_dataloader, optimizer, criterion_coord,
    criterion_angle, device, writer, epochs, scheduler, model_checkpoints
):
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):
        losses = []
        losses_base_point = []
        losses_base_angle = []
        logging.warning(f"Train epoch: {epoch}")
        model.train()
        for step, trajectory in enumerate(train_dataloader):
            logging.warning(f"Train step: {step}")
            h, c = None, None
            for trajectory_step_idx in range(1, trajectory["points"].shape[1]):
                logging.warning(f"trajectory_step_idx: {trajectory_step_idx}")
                masks = (trajectory["masks"][:, trajectory_step_idx] == 1.0).view(-1)
                if int(sum(masks)) == 0:
                    break
                optimizer.zero_grad()
                #with torch.cuda.amp.autocast(enabled=True):
                predict_points = model(
                    trajectory["points"][:, trajectory_step_idx].to(device),
                ) # Maybe use `:trajectory_step_idx` instead `trajectory_step_idx` to train on all history
                #print(predict_points.shape, flush=True)
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
                #h = h.detach()
                #c = c.detach()
                #scaler.scale(loss).backward()
                #scaler.step(optimizer)
                #scaler.update()
                loss.backward()
                optimizer.step()
                #optimizer.zero_grad()

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
        val_ff(
            model, val_dataloader, criterion_coord, criterion_angle,
            device, writer, epoch
        )
        logging.warning(f"Train loss is: {sum(losses)/len(losses)}")
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
        )
