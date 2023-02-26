#!/usr/bin/env python3
import copy
import math
import time

from sweetie_bot_control_msgs.msg import FollowStepSequenceActionGoal

from message_templates import BASE_MOTION_TEMPLATE, LEG_TEMPLATE
from message_templates import MESSAGE_ORDER, LEGS_ORDER
from main_utils import remove_state_part_from_trajectory_point

def create_message(trajectory_points):
    message = FollowStepSequenceActionGoal()
    ts = time.time()
    message.header.stamp.secs = math.floor(ts)
    message.header.stamp.nsecs = math.floor(
        (ts - math.floor(ts)) * 10 ** 7
    )
    message.goal_id.stamp.secs = message.header.stamp.secs
    message.goal_id.stamp.nsecs = message.header.stamp.nsecs + 200
    message.goal_id.id = (
        f"/clop_generator-1-{message.goal_id.stamp.secs}."
        f"{message.goal_id.stamp.nsecs}"
    )
    message.goal.header.frame_id = "odom_combined"
    message.goal.header.seq = 0
    message.goal.header.stamp.secs = message.header.stamp.secs
    message.goal.header.stamp.nsecs = message.header.stamp.nsecs
    message.goal.append = False
    message.goal.position_tolerance = 0.07
    message.goal.orientation_tolerance = 0.5

    points_len = len(trajectory_points)
    time_from_start = [0]
    for idx in range(points_len - 1):
        time_from_start.append(time_from_start[-1] + 0.056)
    message.goal.time_from_start = time_from_start

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
    for step, point in enumerate(trajectory_points[:3]):
        cur_point = remove_state_part_from_trajectory_point(point)
        step = { x: copy.deepcopy(LEG_TEMPLATE) for x in LEGS_ORDER }
        step["base_motion"] = copy.deepcopy(BASE_MOTION_TEMPLATE)
        for k, v in zip(MESSAGE_ORDER, cur_point):
            step[k[0]][k[1]][k[2]][k[3]] = v
        steps["base_motion"]["points"].append(step["base_motion"])
        for idx in range(len(LEGS_ORDER)):
            steps["ee_motion"][idx]["points"].append(step[LEGS_ORDER[idx]])
    message.goal.base_motion = steps["base_motion"]
    message.goal.ee_motion = steps["ee_motion"]
    return None
    return message