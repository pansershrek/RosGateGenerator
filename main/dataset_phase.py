#!/usr/bin/env python3
import os
from typing import Optional
import json
import pickle
import yaml
import logging
import random

import torch
from torch.utils.data import Dataset
import tf_conversions as tfconv
from geometry_msgs.msg import Pose
from PyKDL import *

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor
from main_utils import LRUCache, create_full_trajectoy_point, get_point_from_trajectory_by_idx
from main_utils import get_trajectory_phrame


class MyDatasetFull(Dataset):

    def __init__(self,
        trajectories_path: str,
        generations_path: str,
        trajectory_max_len: Optional[int] = 92,
        cache_size: int = 1000
    ) -> None:
        self.trajectory_max_len = trajectory_max_len
        self.idx2data = {}
        for x in sorted(os.listdir(trajectories_path)):
            self.idx2data[len(self.idx2data)] = {
                "trajectory": os.path.join(trajectories_path, x),
                "generation": os.path.join(
                    generations_path,
                    f"generation_logs_file_{x.split('_')[-1]}"
                )
            }
        self.cacher = LRUCache(cache_size=cache_size)

    def __len__(self) -> int:
        return len(self.idx2data)
        #return len(self.idx2data) * 4 # Each trajectory consists of 4 steps and 2 stend

    def __getitem__(self, idx: int):

        trajectory_idx = idx // 4
        pose_idx = idx % 4
        trajectory_idx = idx
        cache, exist = self.cacher.get(trajectory_idx)
        if exist:
            (trajectory, generation) = cache
        else:
            try:
                with open(self.idx2data[trajectory_idx]["trajectory"], "rb") as f:
                    trajectory_raw = pickle.load(f)
                    trajectory = trajectory_raw
                with open(self.idx2data[trajectory_idx]["generation"], "r") as f:
                    generation = json.load(f)
            except Exception as e:
                logging.warning(f"Invalid trajectory with exception: {e}")
                random_item = random.randint(0, self.__len__() - 1)
                return self.__getitem__(random_item)

            if generation["0"]["error_code"] != 0:
                random_item = random.randint(0, self.__len__() - 1)
                logging.error("Invalid trajectory")
                return self.__getitem__(random_item)

            self.cacher.set(trajectory_idx, (trajectory, generation))

        data = {
            "trajectory_idx": trajectory_idx,
            "pose_idx": pose_idx,
            "shift": create_shift_tesor(generation["0"]["shift"]),
            "features": [],
            "targets": [],
            "mask": 1,
        }

        features_idx = 6 + pose_idx * 20
        features_idx = 0
        #print(trajectory.goal.base_motion.points[6])
        #import yaml
        #a = yaml.load(str(trajectory.goal.ee_motion[3].points[6]), Loader=yaml.CBaseLoader)
        #print(json.dumps(a, indent =4))
        data["features"] = data["shift"] + get_point_from_trajectory_by_idx(
            trajectory, features_idx
        )
        data["targets"] = get_trajectory_phrame(trajectory, features_idx)

        data["shift"] = torch.FloatTensor(data["shift"])
        data["features"] = torch.FloatTensor(data["features"])
        data["targets"] = torch.FloatTensor(data["targets"])
        return data