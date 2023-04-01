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
        return len(self.idx2data) * self.trajectory_max_len

    def __getitem__(self, idx: int):

        trajectory_idx = idx // self.trajectory_max_len
        pose_idx = idx % self.trajectory_max_len
        cache, exist = self.cacher.get(trajectory_idx)
        if exist:
            (trajectory, generation) = cache
        else:
            try:
                with open(self.idx2data[trajectory_idx]["trajectory"], "rb") as f:
                    trajectory_raw = pickle.load(f)
                    trajectory = trajectory_raw
                    #trajectory = yaml.safe_load(str(trajectory_raw))
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
            "features": [0 for _ in range(35)],
            "targets": [0 for _ in range(35)],
            "mask": 0,
        }

        if pose_idx >= len(trajectory.goal.base_motion.points) - 1:
            return data

        f_base_motion = trajectory.goal.base_motion.points[pose_idx]
        t_base_motion = trajectory.goal.base_motion.points[pose_idx + 1]


        T0 = tfconv.fromMsg(f_base_motion.pose)
        T1 = tfconv.fromMsg(t_base_motion.pose)

        T0.p[2] = 0
        T0inv = T0.Inverse()

        f_base_motion.pose = tfconv.toMsg(T0inv * T0)
        t_base_motion.pose = tfconv.toMsg(T0inv * T1)

        T0_twist = Twist(
            Vector(
                f_base_motion.twist.linear.x,
                f_base_motion.twist.linear.y,
                f_base_motion.twist.linear.z
            ),
            Vector(
                f_base_motion.twist.angular.x,
                f_base_motion.twist.angular.y,
                f_base_motion.twist.angular.z
            )
        )
        T1_twist = Twist(
            Vector(
                t_base_motion.twist.linear.x,
                t_base_motion.twist.linear.y,
                t_base_motion.twist.linear.z
            ),
            Vector(
                t_base_motion.twist.angular.x,
                t_base_motion.twist.angular.y,
                t_base_motion.twist.angular.z
            )
        )

        T0_twist = T0inv * T0_twist
        T1_twist = T0inv * T1_twist

        f_base_motion.twist.linear.x = T0_twist.vel[0]
        f_base_motion.twist.linear.y = T0_twist.vel[1]
        f_base_motion.twist.linear.z = T0_twist.vel[2]
        f_base_motion.twist.angular.x = T0_twist.rot[3]
        f_base_motion.twist.angular.y = T0_twist.rot[4]
        f_base_motion.twist.angular.z = T0_twist.rot[5]

        t_base_motion.twist.linear.x = T1_twist.vel[0]
        t_base_motion.twist.linear.y = T1_twist.vel[1]
        t_base_motion.twist.linear.z = T1_twist.vel[2]
        t_base_motion.twist.angular.x = T1_twist.rot[3]
        t_base_motion.twist.angular.y = T1_twist.rot[4]
        t_base_motion.twist.angular.z = T1_twist.rot[5]



        #print(T0_twist[0])
        #print(dir(T0_twist))
        #print(T0_twist.rot)

        return data

        data = {
            "trajectory_idx": idx,
            "shift": create_shift_tesor(generation["0"]["shift"]),
            "points": [],
            "masks": []
        }

        print(trajectory)

        points = []
        for x in trajectory["goal"]["base_motion"]["points"]:
            points.append(
                {
                    "step": len(points),
                    "base_motion": x
                }
            )
            # Remove constant parameters
            points[-1]["base_motion"].pop("accel", None)
            points[-1]["base_motion"]["pose"]["position"].pop("z", None)
            points[-1]["base_motion"]["pose"]["orientation"].pop("x", None)
            points[-1]["base_motion"]["pose"]["orientation"].pop("y", None)
            points[-1]["base_motion"]["twist"]["linear"].pop("z", None)
            points[-1]["base_motion"]["twist"]["angular"].pop("x", None)
            points[-1]["base_motion"]["twist"]["angular"].pop("y", None)

        for leg_position in trajectory["goal"]["ee_motion"]:
            for idx_tmp, x in enumerate(leg_position["points"]):
                points[idx_tmp][leg_position["name"]] = x

                # Remove constant parameters
                points[idx_tmp][leg_position["name"]].pop("accel", None)
                points[idx_tmp][leg_position["name"]]["pose"].pop(
                    "orientation", None
                )
                points[idx_tmp][leg_position["name"]]["twist"]["angular"].pop(
                    "x", None
                )
                points[idx_tmp][leg_position["name"]]["twist"]["angular"].pop(
                    "y", None
                )

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

        return data