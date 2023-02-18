#!/usr/bin/env python3
import json
import copy
from collections import OrderedDict
import random

import torch
import numpy as np

def pprint_trajctory_point(point: dict) -> None:
    print(json.dumps(point, indent=4))

def create_tensor_from_trajectory_point(point: dict) -> list:
    coord_tensor = []
    leg_contacs = [] #leg1, leg2, leg3, leg4
    tmp_point = copy.deepcopy(point)
    tmp_point["base_motion"].pop("contact", None)
    for leg in ["leg1", "leg2", "leg3", "leg4"]:
        leg_contacs.append(tmp_point[leg].pop("contact", False))

    for x in sorted(tmp_point.keys()):
        if x == "step":
            continue
        for y in sorted(tmp_point[x].keys()):
            for z in sorted(tmp_point[x][y].keys()):
                for k in sorted(tmp_point[x][y][z].keys()):
                    coord_tensor.append(tmp_point[x][y][z][k])
    print(len(coord_tensor), len(leg_contacs))
    return torch.FloatTensor(coord_tensor), torch.FloatTensor(leg_contacs)

def create_shift_tesor(shift: dict):
    return torch.FloatTensor([shift["x"], shift["y"], shift["angle"]])

def setup_seed(seed: int=1717) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_device(config: dict) -> dict:
    if config["DEVICE"] == "cuda":
        config["DEVICE"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        config["DEVICE"] = torch.device("cpu")
    return config

def get_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return setup_device(config)

