#!/usr/bin/env python3
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR

from train import train
from dataset import MyDataset
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


    train_dataset = MyDataset(config["TRAIN_TRAJECTORY_PATH"], config["TRAIN_GENERATION_PATH"])

    model = MyModel(95+4, 256, 3)
    model = model.to(config["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=config["LR"])

    loss = MSELoss()
    scheduler = StepLR(optimizer, step_size=len(train_dataset)*3, gamma=0.85)

    train(
        model, train_dataset, None, loss, optimizer, config["DEVICE"],
        writer, config["EPOCHS"], scheduler
    )



if __name__ == "__main__":
    main()