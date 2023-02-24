#!/usr/bin/env python3
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import MSELoss, BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from train import train
from dataset_full import MyDatasetFull
from model import MyModel
from main_utils import get_config, setup_seed

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
        cache_size=300
    )
    val_dataset = MyDatasetFull(
        config["TEST_TRAJECTORY_PATH"],
        config["TEST_GENERATION_PATH"],
        trajectory_max_len = 2369,
        cache_size=300
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config["BATCH_SIZE"],
        shuffle = True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = config["BATCH_SIZE"],
        shuffle = False
    )

    model = MyModel(2 * 35 + 3, 35, 256, 3)
    model = model.to(config["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=config["LR"]) # Maybe use LBFGS???

    loss_coord = MSELoss()
    loss_contacs = BCELoss()
    scheduler = StepLR(optimizer, step_size=len(train_dataset)*3, gamma=0.85)

    train(
        model, train_dataloader, val_dataloader, optimizer, loss_coord,
        config["DEVICE"], writer, config["EPOCHS"], scheduler,
        config["MODEL_CHECKPOINTS"]
    )

    #train_all_trajectories(
    #    model, train_dataset, val_dataset, optimizer, loss_coord, loss_contacs,
    #    config["DEVICE"], writer, config["EPOCHS"], scheduler, config["MODEL_CHECKPOINTS"]
    #)



if __name__ == "__main__":
    main()