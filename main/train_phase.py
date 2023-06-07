#!/usr/bin/env python3
import os
import logging

import torch

from main_utils import quaternion_to_euler_torch, get_pred_base_points_and_angle
from val_phase import val_phase

def train_phase(
    model, train_dataloader, val_dataloader, optimizer, criterion,
    device, writer, epochs, scheduler, model_checkpoints
):
    with open("diff", "w") as f:
        pass
    for epoch in range(epochs):
        losses = []
        logging.warning(f"Train epoch: {epoch}")
        model.train()
        for step, data in enumerate(train_dataloader):
            logging.warning(f"Train step: {step}")

            optimizer.zero_grad()
            output = model(
                data["features"].to(device),
            )
            idx_legs = torch.tensor([9, 16, 23, 30, 44, 51, 58, 65, 79, 86, 93, 100, 114, 121, 128, 135, 149, 156, 163, 170, 184, 191, 198, 205, 219, 226, 233, 240, 254, 261, 268, 275, 289, 296, 303, 310, 324, 331, 338, 345, 359, 366, 373, 380, 394, 401, 408, 415, 429, 436, 443, 450, 464, 471, 478, 485, 499, 506, 513, 520, 534, 541, 548, 555, 569, 576, 583, 590, 604, 611, 618, 625, 639, 646, 653, 660])
            loss = criterion(
                output,
                data["targets"].to(device)
            ) #+ 10 * criterion(
            #    output[:, idx_legs],
            #    data["targets"][:, idx_legs].to(device)
            #)
            #print(data["targets"].shape)
            #print("KEK", data["targets"][:, idx_legs][8])
            with open("diff", "a") as f:
                print(f"Epoch {epoch}", file=f)
                print(data["targets"].tolist()[0], file=f)
                print(output.tolist()[0], file=f)
                print((data["targets"].to(device) - output).tolist()[0], file=f)
                print("#"*100, file=f)
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

        val_phase(
            model, val_dataloader, criterion,
            device, writer, epoch
        )
        logging.warning(f"Train loss is: {sum(losses)/len(losses)}")
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoints, f'checkpoint_{epoch}.pt')
        )
