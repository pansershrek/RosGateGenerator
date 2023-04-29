#!/usr/bin/env python3
import json
import copy
from collections import OrderedDict, deque
import random

from scipy.spatial.transform import Rotation
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
        tmp_point[leg].pop("contact", False)
        leg_contacs.append(tmp_point[leg].pop("contact", False))

    for x in sorted(tmp_point.keys()):
        if x == "step":
            continue
        for y in sorted(tmp_point[x].keys()):
            for z in sorted(tmp_point[x][y].keys()):
                for k in sorted(tmp_point[x][y][z].keys()):
                    coord_tensor.append(float(tmp_point[x][y][z][k]))
    return coord_tensor, leg_contacs

def create_tensor_from_trajectory_point_for_predict(point: dict) -> list:
    tmp_point = copy.deepcopy(point)

    tmp_point["base_motion"].pop("accel", None)
    tmp_point["base_motion"]["pose"]["position"].pop("z", None)
    tmp_point["base_motion"]["pose"]["orientation"].pop("x", None)
    tmp_point["base_motion"]["pose"]["orientation"].pop("y", None)
    tmp_point["base_motion"]["twist"]["linear"].pop("z", None)
    tmp_point["base_motion"]["twist"]["angular"].pop("x", None)
    tmp_point["base_motion"]["twist"]["angular"].pop("y", None)
    for x in ["leg1", "leg2", "leg3", "leg4"]:
        tmp_point[x].pop("accel", None)
        tmp_point[x]["pose"].pop(
            "orientation", None
        )
        tmp_point[x]["twist"]["angular"].pop(
            "x", None
        )
        tmp_point[x]["twist"]["angular"].pop(
            "y", None
        )
    return create_tensor_from_trajectory_point(tmp_point)

def create_full_trajectoy_point(shift, start_point, cur_point):
    if start_point is not None:
        return (
            shift +
            create_tensor_from_trajectory_point(start_point)[0] +
            create_tensor_from_trajectory_point(cur_point)[0]
        )
    return (
        shift + create_tensor_from_trajectory_point(cur_point)[0]
    )

def remove_state_part_from_trajectory_point(
    point, shift_part = 3, start_point_part = 35
):
    return point[shift_part + start_point_part: ]

def create_shift_tesor(shift: dict):
    shift_rot = Rotation.from_euler("xyz", [0, 0, shift["angle"]], degrees=False)
    return [
        shift["x"], shift["y"],
        shift_rot.as_quat()[2], shift_rot.as_quat()[3]
    ]

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

def get_base_position(point):
    return {
        "x": point[2], # position
        "y": point[3], # position
        "z": point[1], # orientation
        "w": point[0], # orientation
    }

def get_final_base_position(base_position, shift):
    final_position = copy.deepcopy(base_position)

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
    #The mapping from quaternions to rotations is two-to-one, i.e. quaternions q and -q,
    #where -q simply reverses the sign of each component, represent the same spatial rotation.
    shift_rot = Rotation.from_euler("xyz", [0, 0, shift["angle"]], degrees=False)
    base_rot = Rotation.from_quat([0, 0, base_position["z"], base_position["w"]])

    final_rot = shift_rot * base_rot

    final_position["x"] += shift["x"]
    final_position["y"] += shift["y"]
    final_position["z"] = final_rot.as_quat()[2]
    final_position["w"] = final_rot.as_quat()[3]
    return final_position

def base_distance(cur_base_position, fin_base_position):
    dist = {}
    for key in cur_base_position.keys():
        if key == "x" or key == "y":
            dist[key] = abs(cur_base_position[key] - fin_base_position[key])

    cur_angle = Rotation.from_quat([0, 0, cur_base_position["z"], cur_base_position["w"]])
    fin_angle = Rotation.from_quat([0, 0, fin_base_position["z"], fin_base_position["w"]])

    cur_angle = cur_angle.as_euler("xyz", degrees=False)[2]
    fin_angle = fin_angle.as_euler("xyz", degrees=False)[2]
    dist["angle"] = abs(cur_angle - fin_angle)
    return dist

def is_close(dist, eps):
    for k in dist.keys():
        if dist[k] > eps[k]:
            return False
    return True

def quaternion_to_euler_torch(angle):
    siny_cosp = 2 * (angle[:, 0] * angle[:,1])
    cosy_cosp = 1 - 2 * (angle[:,1] * angle[:,1])
    return torch.atan2(siny_cosp, cosy_cosp)

def get_pred_base_points_and_angle(predict_points, masks):
    pred_base_points = predict_points[masks, :4]
    if len(pred_base_points.shape) == 1:
        pred_base_angle = pred_base_points[:2]
        pred_base_points = pred_base_points[2:]
    else:
        pred_base_angle = pred_base_points[:, :2]
        pred_base_points = pred_base_points[:, 2:]

    return pred_base_points, pred_base_angle

def get_point_from_trajectory_by_idx(trajectory, idx):
    ans = [
        # Base motion
        trajectory.goal.base_motion.points[idx].pose.orientation.w,
        trajectory.goal.base_motion.points[idx].pose.orientation.z,
        trajectory.goal.base_motion.points[idx].pose.position.x,
        trajectory.goal.base_motion.points[idx].pose.position.y,
        trajectory.goal.base_motion.points[idx].twist.angular.z,
        trajectory.goal.base_motion.points[idx].twist.linear.x,
        trajectory.goal.base_motion.points[idx].twist.linear.y,
        # Leg1
        trajectory.goal.ee_motion[0].points[idx].pose.position.x,
        trajectory.goal.ee_motion[0].points[idx].pose.position.y,
        trajectory.goal.ee_motion[0].points[idx].pose.position.z,
        trajectory.goal.ee_motion[0].points[idx].twist.angular.z,
        trajectory.goal.ee_motion[0].points[idx].twist.linear.x,
        trajectory.goal.ee_motion[0].points[idx].twist.linear.y,
        trajectory.goal.ee_motion[0].points[idx].twist.linear.z,
        # Leg2
        trajectory.goal.ee_motion[1].points[idx].pose.position.x,
        trajectory.goal.ee_motion[1].points[idx].pose.position.y,
        trajectory.goal.ee_motion[1].points[idx].pose.position.z,
        trajectory.goal.ee_motion[1].points[idx].twist.angular.z,
        trajectory.goal.ee_motion[1].points[idx].twist.linear.x,
        trajectory.goal.ee_motion[1].points[idx].twist.linear.y,
        trajectory.goal.ee_motion[1].points[idx].twist.linear.z,
        # Leg3
        trajectory.goal.ee_motion[2].points[idx].pose.position.x,
        trajectory.goal.ee_motion[2].points[idx].pose.position.y,
        trajectory.goal.ee_motion[2].points[idx].pose.position.z,
        trajectory.goal.ee_motion[2].points[idx].twist.angular.z,
        trajectory.goal.ee_motion[2].points[idx].twist.linear.x,
        trajectory.goal.ee_motion[2].points[idx].twist.linear.y,
        trajectory.goal.ee_motion[2].points[idx].twist.linear.z,
        # Leg4
        trajectory.goal.ee_motion[3].points[idx].pose.position.x,
        trajectory.goal.ee_motion[3].points[idx].pose.position.y,
        trajectory.goal.ee_motion[3].points[idx].pose.position.z,
        trajectory.goal.ee_motion[3].points[idx].twist.angular.z,
        trajectory.goal.ee_motion[3].points[idx].twist.linear.x,
        trajectory.goal.ee_motion[3].points[idx].twist.linear.y,
        trajectory.goal.ee_motion[3].points[idx].twist.linear.z,
    ]
    return ans

def get_trajectory_phrame(trajectory, idx):
    ans = []
    #for phrame_idx in range(idx + 1, idx + 20):
    for phrame_idx in range(idx + 1, idx + 6):
        tmp_point = get_point_from_trajectory_by_idx(
            trajectory, phrame_idx
        )
        ans += tmp_point
        #ans += (
        #    tmp_point[: 7] +
        #    tmp_point[7: 9] +
        #    tmp_point[14: 16] +
        #    tmp_point[21: 23] +
        #    tmp_point[28: 30]
        #) # Do we have to predict only x,y or x,y,z of a leg?
    return ans
