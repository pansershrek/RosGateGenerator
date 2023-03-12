#!/usr/bin/env python3
import argparse
import traceback
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import MSELoss, BCELoss, L1Loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from train import train
from dataset_full import MyDatasetFull
from model import MyModel
from main_utils import get_config, setup_seed
from create_message import create_message
from inference import inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Config file path",
        default="/home/panser/Desktop/RosGateGenerator/main/config.json",
    )
    args = parser.parse_args()

    config = get_config(args.config_path)
    setup_seed(config["SEED"])

    writer = SummaryWriter(log_dir=config["LOG_PATH"])

    train_dataset = MyDatasetFull(
        config["TRAIN_TRAJECTORY_PATH"],
        config["TRAIN_GENERATION_PATH"],
        trajectory_max_len = 2369,
        cache_size=1000
    )
    val_dataset = MyDatasetFull(
        config["TEST_TRAJECTORY_PATH"],
        config["TEST_GENERATION_PATH"],
        trajectory_max_len = 2369,
        cache_size=1000
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config["BATCH_SIZE"],
        shuffle = True,
        num_workers = 16,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = config["BATCH_SIZE"],
        shuffle = False,
        num_workers = 16,
    )

    model = MyModel(2 * 35 + 3, 35, 256, 5)
    model = model.to(config["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=config["LR"]) # Maybe use LBFGS???

    loss_coord = L1Loss()
    loss_angle = MSELoss()
    loss_contacs = BCELoss()
    scheduler = StepLR(optimizer, step_size=len(train_dataset) * 3, gamma=0.95)

    if config["MODE"] == "INFERENCE":
        #model.load_state_dict(torch.load(config["INFERENCE_MODEL_PATH"]))
        model.eval()
        inference_points = inference(
            model, config["INFERENCE_START_POINT"],
            config["INFERENCE_SHIFT"], config["DEVICE"]
        )
        ros_message = create_message(inference_points)
        with open(config["INFERENCE_TRAJECTORY_PATH"], "wb") as f:
            pickle.dump(ros_message, f)
    else:
        try:
            train(
                model, train_dataloader, val_dataloader, optimizer,
                loss_coord, loss_angle, config["DEVICE"], writer,
                config["EPOCHS"], scheduler, config["MODEL_CHECKPOINTS"]
            )
        except Exception as e:
            print(f"Exception {e}")
            print(f"Traceback {traceback.format_exc()}")




if __name__ == "__main__":
    main()