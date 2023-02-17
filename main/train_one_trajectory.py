#!/usr/bin/env python3

from main_utils import create_tensor_from_trajectory_point

def train_one_trajectory(
    model, trajectory, criterion, optimizer, device,
    writer, epochs, scheduler
)-> None:
    #model.train()
    for point in trajectory["points"]:
        coords_tensor, contact_tensor = create_tensor_from_trajectory_point(
            point
        )
        return None