#!/usr/bin/env python3
import logging

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor
from main_utils import quaternion_to_euler_torch, get_pred_base_points_and_angle

def val_phase(
    model, val_dataloader, criterion,
    device, writer, epoch
):
    model.eval()
    losses = []
    logging.warning(f"Val epoch: {epoch}")
    for step, data in enumerate(val_dataloader):
        output = model(
            data["features"].to(device),
        )
        loss = criterion(
            output,
            data["targets"].to(device)
        )
        losses.append(loss.item())
    logging.warning(f"Val loss is: {sum(losses)/len(losses)}")
    if writer is not None:
        writer.add_scalar(
            "val/loss",
            sum(losses)/len(losses), epoch
        )
