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

from main_utils import remove_state_part_from_trajectory_point
from message_templates import BASE_MOTION_TEMPLATE, LEG_TEMPLATE
from message_templates import MESSAGE_ORDER, LEGS_ORDER
from leg_contact_const import Stand, Walk2, Walk2E
import copy


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
        #return 1
        #return len(self.idx2data)
        return len(self.idx2data) * 4 # Each trajectory consists of 4 steps and 2 stend

    def __getitem__(self, _idx: int):
        #idx = 8
        #idx = 2
        trajectory_idx = _idx // 4
        pose_idx = _idx % 4
        #trajectory_idx = idx
        #print(_idx, self.idx2data[trajectory_idx])
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

        #trajectory.goal.base_motion.points = trajectory.goal.base_motion.points[6:26]
        #trajectory.goal.ee_motion[0].points = trajectory.goal.ee_motion[0].points[6:26]
        #trajectory.goal.ee_motion[1].points = trajectory.goal.ee_motion[1].points[6:26]
        #trajectory.goal.ee_motion[2].points = trajectory.goal.ee_motion[2].points[6:26]
        #trajectory.goal.ee_motion[3].points = trajectory.goal.ee_motion[3].points[6:26]
        #with open("original_trajectory.yaml", "w") as f:
        #    print(trajectory.goal.ee_motion[3].points[6:26], file=f)
        data = {
            "trajectory_idx": trajectory_idx,
            "pose_idx": pose_idx,
            "shift": create_shift_tesor(generation["0"]["shift"]),
            "shift_orig": generation["0"]["shift"],
            "features": [],
            "targets": [],
            "mask": 1,
            "idx": _idx
        }

        features_idx = 6 + pose_idx * 20
        if pose_idx > 0:
            features_idx -= 1
        #features_idx = 0
        #print(trajectory.goal.base_motion.points[6])
        #import yaml
        #a = yaml.load(str(trajectory.goal.ee_motion[3].points[6]), Loader=yaml.CBaseLoader)
        #print(json.dumps(a, indent =4))
        data["features"] = data["shift"] + get_point_from_trajectory_by_idx(
            trajectory, features_idx
        )[28:]
        #print("original feature", data["features"])
        data["targets"] = get_trajectory_phrame(trajectory, features_idx)
        #print("original targets", data["targets"])
        data["shift"] = torch.FloatTensor(data["shift"])
        data["features"] = torch.FloatTensor(data["features"])
        data["targets"] = torch.FloatTensor(data["targets"])


        steps = {
            "ee_motion": [
                {
                    "name": x,
                    "points": []
                } for x in LEGS_ORDER
            ]
        }
        steps["base_motion"] = {
            "name": "base_link",
            "points": []
        }
        point = get_point_from_trajectory_by_idx(
            trajectory, features_idx
        )
        point = get_point_from_trajectory_by_idx(trajectory, features_idx)
        step = { x: copy.deepcopy(LEG_TEMPLATE) for x in LEGS_ORDER }
        step["base_motion"] = copy.deepcopy(BASE_MOTION_TEMPLATE)
        for k, v in zip(MESSAGE_ORDER, point):
            step[k[0]][k[1]][k[2]][k[3]] = v
        steps["base_motion"]["points"].append(step["base_motion"])
        for idx in range(len(LEGS_ORDER)):
            steps["ee_motion"][idx]["points"].append(step[LEGS_ORDER[idx]])
        steps["step"] = 0
        for x in steps["ee_motion"]:
            steps[x["name"]] = x["points"][0]
        steps.pop("ee_motion")
        steps["base_motion"] = steps["base_motion"]["points"][0]
        #print(json.dumps(steps, indent=4))
        #if _idx == 8:
        #    print(json.dumps(steps, indent=4))
        return data