#!/usr/bin/env python3
import os
from typing import Optional
import json
import pickle
import yaml
import logging

import torch
from torch.utils.data import Dataset
import tf_conversions
from geometry_msgs.msg import Pose

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor
from main_utils import LRUCache, create_full_trajectoy_point

class MyDatasetFull(Dataset):

    def __init__(self,
        trajectories_path: str,
        generations_path: str,
        trajectory_max_len: Optional[int] = None,
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

    def __getitem__(self, idx: int):

        data, exist = self.cacher.get(idx)
        if exist:
            return data
        try:
            with open(self.idx2data[idx]["trajectory"], "rb") as f:
                trajectory_raw = pickle.load(f)
                #print(trajectory_raw)
                #trajectory = yaml.load(str(trajectory_raw), Loader=yaml.CBaseLoader)
                trajectory = trajectory_raw
            with open(self.idx2data[idx]["generation"], "r") as f:
                generation = json.load(f)
        except Exception as e:
            logging.warning(f"Invalid trajectory with exception: {e}")
            random_item = random.randint(0, len(self.idx2data) - 1)
            return self.__getitem__(random_item)

        if generation["0"]["error_code"] != 0:
            random_item = random.randint(0, len(self.idx2data) - 1)
            logging.error("Invalid trajectory")
            return self.__getitem__(random_item)

        data = {
            "trajectory_idx": idx,
            "shift": create_shift_tesor(generation["0"]["shift"]),
            "points": [],
            "masks": []
        }
        points = []
        for x in trajectory.goal.base_motion.points:
            y = {
                "pose": {
                    "position": {
                        "x": x.pose.position.x,
                        "y": x.pose.position.y,
                    },
                    "orientation": {
                        "z": x.pose.orientation.z,
                        "w": x.pose.orientation.w,
                    }
                },
                "twist": {
                    "linear": {
                        "x": x.twist.linear.x,
                        "y": x.twist.linear.y,
                    },
                    "angular": {
                        "z": x.twist.angular.z,
                    }
                }
            }
            points.append(
                {
                    "step": len(points),
                    "base_motion": y
                }
            )
            # Remove constant parameters
            #points[-1]["base_motion"].pop("accel", None)
            #points[-1]["base_motion"]["pose"]["position"].pop("z", None)
            #points[-1]["base_motion"]["pose"]["orientation"].pop("x", None)
            #points[-1]["base_motion"]["pose"]["orientation"].pop("y", None)
            #points[-1]["base_motion"]["twist"]["linear"].pop("z", None)
            #points[-1]["base_motion"]["twist"]["angular"].pop("x", None)
            #points[-1]["base_motion"]["twist"]["angular"].pop("y", None)

        for leg_position in trajectory.goal.ee_motion:
            for idx_tmp, x in enumerate(leg_position.points):
                y = {
                    "pose": {
                        "position": {
                            "x": x.pose.position.x,
                            "y": x.pose.position.y,
                            "z": x.pose.position.z,
                        }
                    },
                    "twist": {
                        "linear": {
                            "x": x.twist.linear.x,
                            "y": x.twist.linear.y,
                            "z": x.twist.linear.z,
                        },
                        "angular": {
                            "z": x.twist.angular.z,
                        }
                    }
                }
                points[idx_tmp][leg_position.name] = y

                # Remove constant parameters
                #points[idx_tmp][leg_position["name"]].pop("accel", None)
                #points[idx_tmp][leg_position["name"]]["pose"].pop(
                #    "orientation", None
                #)
                #points[idx_tmp][leg_position["name"]]["twist"]["angular"].pop(
                #    "x", None
                #)
                #points[idx_tmp][leg_position["name"]]["twist"]["angular"].pop(
                #    "y", None
                #)

        data["start_point"] = create_tensor_from_trajectory_point(
            points[0]
        )[0]
        data["real_len"] = len(points)

        for idx_tmp, x in enumerate(points):
            if (
                self.trajectory_max_len is not None and
                idx_tmp >= self.trajectory_max_len
            ):
                break
            #data["points"].append(
            #    create_full_trajectoy_point(data["shift"], points[0], x)
            #)
            data["points"].append(
                create_full_trajectoy_point(data["shift"], None, x)
            )
            if (
                self.trajectory_max_len is not None and
                idx_tmp >= self.trajectory_max_len
            ):
                data["masks"].append([0])
            else:
                data["masks"].append([1])

        if self.trajectory_max_len is not None:
            while len(data["points"]) < self.trajectory_max_len:
                data["points"].append(
                    [0 for x in data["points"][-1]]
                )
                data["masks"].append([0])

        data["shift"] = torch.FloatTensor(data["shift"])
        data["points"] = torch.FloatTensor(data["points"])
        data["masks"] = torch.FloatTensor(data["masks"])
        data["start_point"] = torch.FloatTensor(data["start_point"])

        self.cacher.set(idx, data)

        return data