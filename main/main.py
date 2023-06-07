#!/usr/bin/env python3
import argparse
import traceback
import pickle
import json

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
from model import MyModel, MyModelFF, MyModelFF_head
from main_utils import get_config, setup_seed, LRUCache
from create_message import create_message
from inference import inference
from inference_ff import inference_ff
from inference_sep import inference_sep
from change_coord import change_coord

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
    train_dataset[0]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config["BATCH_SIZE"],
        shuffle = False,
        num_workers = 16,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = config["BATCH_SIZE"],
        shuffle = False,
        num_workers = 16,
    )
    #model = MyModelFF(35 + 3, 35, 256, 1)
    #model = MyModelFF(35 + 4, 285, 1024, 5)
    #model = MyModelFF(35 + 4, 665, 1024, 5)
    #model = MyModelFF(35 + 4, 35 * 5, 1024, 5)
    #model = MyModelFF(35 + 4, 133, 1024, 5)
    model = MyModelFF(7 + 4, 133, 1024, 5)
    model = model.to(config["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=config["LR"]) # Maybe use LBFGS???

    loss_coord = MSELoss(reduction="sum") #L1Loss(reduction="sum") #MSELoss(reduction="sum") #L1Loss()
    loss_angle = MSELoss()
    loss_contacs = BCELoss()
    scheduler = StepLR(optimizer, step_size=100, gamma=0.95)

    if config["MODE"] == "INFERENCE":
        model_legs1 = MyModelFF(7 + 4, 133, 1024, 5)
        model_legs1 = model_legs1.to(config["DEVICE"])
        model_legs1.load_state_dict(torch.load("/home/panser/Desktop/RosGateGenerator/main/leg1_model_checkpoints/checkpoint_511.pt"))
        model_legs2 = MyModelFF(7 + 4, 133, 1024, 5)
        model_legs2 = model_legs2.to(config["DEVICE"])
        model_legs2.load_state_dict(torch.load("/home/panser/Desktop/RosGateGenerator/main/leg2_model_checkpoints/checkpoint_511.pt"))
        model_legs3 = MyModelFF(7 + 4, 133, 1024, 5)
        model_legs3 = model_legs3.to(config["DEVICE"])
        model_legs3.load_state_dict(torch.load("/home/panser/Desktop/RosGateGenerator/main/leg3_model_checkpoints/checkpoint_511.pt"))
        model_legs4 = MyModelFF(7 + 4, 133, 1024, 5)
        model_legs4 = model_legs4.to(config["DEVICE"])
        model_legs4.load_state_dict(torch.load("/home/panser/Desktop/RosGateGenerator/main/leg4_model_checkpoints/checkpoint_511.pt"))
        model_head = MyModelFF(7 + 4, 133, 1024, 5)
        model_head = model_head.to(config["DEVICE"])
        model_head.load_state_dict(torch.load("/home/panser/Desktop/RosGateGenerator/main/base_model_checkpoints/checkpoint_511.pt"))
        inference_points = inference_sep(
            model_head, model_legs1, model_legs2, model_legs3, model_legs4, config["INFERENCE_START_POINT"],
            config["INFERENCE_SHIFT"], config["DEVICE"]
        )
        ros_message = create_message(inference_points)

        with open(config["INFERENCE_TRAJECTORY_PATH"], "wb") as f:
            pickle.dump(ros_message, f)
        print(len(ros_message.goal.base_motion.points))
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