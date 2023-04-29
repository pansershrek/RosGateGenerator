#!/usr/bin/env python3
import copy
import math
import time

from sweetie_bot_control_msgs.msg import FollowStepSequenceActionGoal
from sweetie_bot_control_msgs.msg import RigidBodyTrajectory
from sweetie_bot_control_msgs.msg import RigidBodyTrajectoryPoint

from main_utils import remove_state_part_from_trajectory_point
from message_templates import BASE_MOTION_TEMPLATE, LEG_TEMPLATE
from message_templates import MESSAGE_ORDER, LEGS_ORDER
from leg_contact_const import Stand, Walk2, Walk2E

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
    message.goal.header.seq = 1
    message.goal.header.stamp.secs = message.header.stamp.secs
    message.goal.header.stamp.nsecs = message.header.stamp.nsecs
    message.goal.append = False
    message.goal.position_tolerance = 1
    message.goal.orientation_tolerance = 1

    points_len = len(trajectory_points)
    time_from_start = [0]
    for idx in range(points_len - 1):
        time_from_start.append(time_from_start[-1] + 0.0375)
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
    for point in trajectory_points:
        step = { x: copy.deepcopy(LEG_TEMPLATE) for x in LEGS_ORDER }
        step["base_motion"] = copy.deepcopy(BASE_MOTION_TEMPLATE)
        for k, v in zip(MESSAGE_ORDER, point):
            step[k[0]][k[1]][k[2]][k[3]] = v
        steps["base_motion"]["points"].append(step["base_motion"])
        for idx in range(len(LEGS_ORDER)):
            steps["ee_motion"][idx]["points"].append(step[LEGS_ORDER[idx]])

    contact_stack = []
    contact_stack += Stand
    while (
        len(contact_stack) + len(Walk2E) + len(Stand) <
        len(trajectory_points)
    ):
        contact_stack += Walk2
    contact_stack += Walk2E

    while len(contact_stack) < len(trajectory_points):
        contact_stack += Stand

    for idx in range(len(LEGS_ORDER)):
        for point_idx in range(len(steps["ee_motion"][idx]["points"])):
            steps["ee_motion"][idx]["points"][point_idx]["contact"] = (
                contact_stack[point_idx][idx]
            )

    base_motion = RigidBodyTrajectory()
    base_motion.name = "base_link"

    tmp = []
    for x in steps["base_motion"]["points"]:
        base_RigidBodyTrajectoryPoint = RigidBodyTrajectoryPoint()
        base_RigidBodyTrajectoryPoint.contact = x["contact"]
        base_RigidBodyTrajectoryPoint.pose.position.x = x["pose"]["position"]["x"]
        base_RigidBodyTrajectoryPoint.pose.position.y = x["pose"]["position"]["y"]
        base_RigidBodyTrajectoryPoint.pose.position.z = x["pose"]["position"]["z"]
        base_RigidBodyTrajectoryPoint.pose.orientation.x = x["pose"]["orientation"]["x"]
        base_RigidBodyTrajectoryPoint.pose.orientation.y = x["pose"]["orientation"]["y"]
        base_RigidBodyTrajectoryPoint.pose.orientation.z = x["pose"]["orientation"]["z"]
        base_RigidBodyTrajectoryPoint.pose.orientation.w = x["pose"]["orientation"]["w"]
        base_RigidBodyTrajectoryPoint.twist.linear.x = x["twist"]["linear"]["x"]
        base_RigidBodyTrajectoryPoint.twist.linear.y = x["twist"]["linear"]["y"]
        base_RigidBodyTrajectoryPoint.twist.linear.z = x["twist"]["linear"]["z"]
        base_RigidBodyTrajectoryPoint.twist.angular.x = x["twist"]["angular"]["x"]
        base_RigidBodyTrajectoryPoint.twist.angular.y = x["twist"]["angular"]["y"]
        base_RigidBodyTrajectoryPoint.twist.angular.z = x["twist"]["angular"]["z"]
        base_RigidBodyTrajectoryPoint.accel.linear.x = x["accel"]["linear"]["x"]
        base_RigidBodyTrajectoryPoint.accel.linear.y = x["accel"]["linear"]["y"]
        base_RigidBodyTrajectoryPoint.accel.linear.z = x["accel"]["linear"]["z"]
        base_RigidBodyTrajectoryPoint.accel.angular.x = x["accel"]["angular"]["x"]
        base_RigidBodyTrajectoryPoint.accel.angular.y = x["accel"]["angular"]["y"]
        base_RigidBodyTrajectoryPoint.accel.angular.z = x["accel"]["angular"]["z"]
        tmp.append(base_RigidBodyTrajectoryPoint)
    base_motion.points = tmp
    message.goal.base_motion = base_motion

    all_legs = []
    for leg, leg_name in zip(steps["ee_motion"], LEGS_ORDER):
        ee_motion = RigidBodyTrajectory()
        ee_motion.name = leg_name
        tmp = []
        for x in leg["points"]:
            leg_RigidBodyTrajectoryPoint = RigidBodyTrajectoryPoint()
            leg_RigidBodyTrajectoryPoint.contact = x["contact"]
            leg_RigidBodyTrajectoryPoint.pose.position.x = x["pose"]["position"]["x"]
            leg_RigidBodyTrajectoryPoint.pose.position.y = x["pose"]["position"]["y"]
            leg_RigidBodyTrajectoryPoint.pose.position.z = x["pose"]["position"]["z"]
            leg_RigidBodyTrajectoryPoint.pose.orientation.x = x["pose"]["orientation"]["x"]
            leg_RigidBodyTrajectoryPoint.pose.orientation.y = x["pose"]["orientation"]["y"]
            leg_RigidBodyTrajectoryPoint.pose.orientation.z = x["pose"]["orientation"]["z"]
            leg_RigidBodyTrajectoryPoint.pose.orientation.w = x["pose"]["orientation"]["w"]
            leg_RigidBodyTrajectoryPoint.twist.linear.x = x["twist"]["linear"]["x"]
            leg_RigidBodyTrajectoryPoint.twist.linear.y = x["twist"]["linear"]["y"]
            leg_RigidBodyTrajectoryPoint.twist.linear.z = x["twist"]["linear"]["z"]
            leg_RigidBodyTrajectoryPoint.twist.angular.x = x["twist"]["angular"]["x"]
            leg_RigidBodyTrajectoryPoint.twist.angular.y = x["twist"]["angular"]["y"]
            leg_RigidBodyTrajectoryPoint.twist.angular.z = x["twist"]["angular"]["z"]
            leg_RigidBodyTrajectoryPoint.accel.linear.x = x["accel"]["linear"]["x"]
            leg_RigidBodyTrajectoryPoint.accel.linear.y = x["accel"]["linear"]["y"]
            leg_RigidBodyTrajectoryPoint.accel.linear.z = x["accel"]["linear"]["z"]
            leg_RigidBodyTrajectoryPoint.accel.angular.x = x["accel"]["angular"]["x"]
            leg_RigidBodyTrajectoryPoint.accel.angular.y = x["accel"]["angular"]["y"]
            leg_RigidBodyTrajectoryPoint.accel.angular.z = x["accel"]["angular"]["z"]
            tmp.append(leg_RigidBodyTrajectoryPoint)
        ee_motion.points = tmp
        all_legs.append(ee_motion)

    message.goal.ee_motion = all_legs

    return message