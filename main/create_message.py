#!/usr/bin/env python3
import math
import time

from sweetie_bot_control_msgs.msg import FollowStepSequenceActionGoal

from message_order import MESSAGE_ORDER

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
    message.goal.base_motion.name = "base_link"

    #for point in trajectory_points:
    #    print(point)
    #    break
    return message