#!/usr/bin/env python3
import json
import copy
from collections import OrderedDict, deque
import random

import torch
import numpy as np

class LRUCache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = deque()
        self.hash_map = dict()

    def is_queue_full(self):
        return len(self.queue) == self.cache_size

    def set(self, key, value):
        if self.cache_size in [0, None]:
            return
        if key not in self.hash_map:
            if self.is_queue_full():
                pop_key = self.queue.pop()
                self.hash_map.pop(pop_key)
                self.queue.appendleft(key)
                self.hash_map[key] = value
            else:
                self.queue.appendleft(key)
                self.hash_map[key] = value

    def get(self, key):
        if key not in self.hash_map:
            return -1, False
        else:
            self.queue.remove(key)
            self.queue.appendleft(key)
            return self.hash_map[key], True

    def clear(self):
        self.queue = deque()
        self.hash_map = dict()

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
    return coord_tensor, leg_contacs

def create_shift_tesor(shift: dict):
    return [shift["x"], shift["y"], shift["angle"]]

def setup_seed(seed: int=1717) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_device(config: dict) -> dict:
    if config["DEVICE"] == "cuda":
        config["DEVICE"] = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        config["DEVICE"] = torch.device("cpu")
    return config

def get_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return setup_device(config)

def get_final_point(start_point, shift):
    raise NotImplementedError
