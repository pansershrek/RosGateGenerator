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
from train_ff import train_ff
from train_phase import train_phase
#from dataset_way_coord_system import MyDatasetFull
#from dataset_full import MyDatasetFull
from dataset_phase import MyDatasetFull
from model import MyModel, MyModelFF
from main_utils import get_config, setup_seed, LRUCache
from create_message import create_message
from inference import inference
from inference_ff import inference_ff

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
        trajectory_max_len = 92,
        cache_size=1000
    )
    val_dataset = MyDatasetFull(
        config["TEST_TRAJECTORY_PATH"],
        config["TEST_GENERATION_PATH"],
        trajectory_max_len = 92,
        cache_size=1000
    )
    #train_dataset[4]
    #print(train_dataset[4])
    #return 0
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
    #model = MyModel(2 * 35 + 3, 35, 256, 1)
    #model = MyModelFF(35 + 3, 35, 256, 1)
    #model = MyModelFF(35 + 4, 285, 1024, 5)
    #model = MyModelFF(35 + 4, 665, 1024, 5)
    model = MyModelFF(35 + 4, 35 * 5, 1024, 5)
    model = model.to(config["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=config["LR"]) # Maybe use LBFGS???

    loss_coord = L1Loss()
    loss_angle = MSELoss()
    loss_contacs = BCELoss()
    scheduler = StepLR(optimizer, step_size=4, gamma=0.95)

    if config["MODE"] == "INFERENCE":
        model.load_state_dict(torch.load(config["INFERENCE_MODEL_PATH"]))
        inference_points = inference_ff(
            model, config["INFERENCE_START_POINT"],
            config["INFERENCE_SHIFT"], config["DEVICE"]
        )
        ros_message = create_message(inference_points)
        with open(config["INFERENCE_TRAJECTORY_PATH"], "wb") as f:
            pickle.dump(ros_message, f)
        #with open("my_kek.yaml", "w") as f:
        #    print(ros_message, file=f)
    else:
        try:
            train_phase(
                model, train_dataloader, val_dataloader, optimizer,
                loss_coord, config["DEVICE"], writer,
                config["EPOCHS"], scheduler, config["MODEL_CHECKPOINTS"]
            )
        except Exception as e:
            print(f"Exception {e}")
            print(f"Traceback {traceback.format_exc()}")




if __name__ == "__main__":
    main()