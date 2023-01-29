#!/usr/bin/env python3
import argparse
import json
from math import pi
import random
import sys
import time
import yaml
import logging
from functools import partial

import rospy, actionlib
from std_msgs.msg import String

from flexbe_msgs.msg import BehaviorExecutionAction, BehaviorExecutionGoal
from sweetie_bot_clop_generator.clopper import MoveBaseGoal, MoveBaseAction
from sweetie_bot_control_msgs.msg import FollowStepSequenceActionGoal


class ROSFlexbeTimeoutException(Exception):
    pass


def callback(data, file):
    with open(file, "a") as f:
        print(json.dumps(yaml.safe_load(str(data))), file=f)

def create_listener(trajectory_logs_file):

    rospy.Subscriber(
        "/motion/controller/step_sequence/goal",
        FollowStepSequenceActionGoal,
        partial(callback, file=trajectory_logs_file)
    )

def set_base_into_pose(pose: str) -> str:
    goal = BehaviorExecutionGoal()
    goal.behavior_name = "ExecuteSetPose"
    goal.arg_keys = ["pose", "set_supports"]
    goal.arg_values = [pose, "True"]

    client = actionlib.SimpleActionClient(
        'flexbe/flexbe/execute_behavior', BehaviorExecutionAction
    )
    if not client.wait_for_server(timeout=rospy.Duration(3.0)):
        raise ROSFlexbeTimeoutException(
            'flexbe/flexbe/execute_behavior action is not available.'
        )

    client.send_goal(goal)
    client.wait_for_result()

    return client.get_result()

def set_base_link_height(height: float) -> str:
    goal = BehaviorExecutionGoal()
    goal.behavior_name = "ExecuteSetStance"
    goal.arg_keys.append("target_height")
    goal.arg_values.append(f"{height}")

    # connect to flexbe behavior server
    client = actionlib.SimpleActionClient(
        'flexbe/flexbe/execute_behavior', BehaviorExecutionAction
    )
    if not client.wait_for_server(timeout=rospy.Duration(3.0)):
        raise ROSFlexbeTimeoutException(
            'flexbe/flexbe/execute_behavior action is not available.'
        )

    client.send_goal(goal)
    client.wait_for_result()

    return client.get_result()

def walk_to_coord(x: float = 0.4, y: float = 0.0, angle: float = 0.0) -> str:
    msg = MoveBaseGoal(gait_type = "walk_overlap", n_steps = 4, duration = 3.4)
    msg.setTargetBaseShift(x = x, y = y, angle = angle)
    msg.addEndEffectorsTargets(["leg1","leg2","leg3","leg4"])

    client = actionlib.SimpleActionClient(
        'clop_generator', MoveBaseAction
    )
    if not client.wait_for_server(timeout=rospy.Duration(3.0)):
        raise ROSFlexbeTimeoutException(
            'clop_generator action is not available.'
        )

    client.send_goal(msg)
    client.wait_for_result()

    return client.get_result()

def get_random_shift() -> float:
    # Max shift for 4 steps in X way is 0.5 and in Y way is 0.4
    shift = random.randint(0, 4) / 10.0
    sign = random.randint(0, 1)
    return sign * shift - (1 - sign) * shift

def get_random_angle() -> float:
    # Robot can turn in any angle if robot makes only this movement
    angle = random.randint(0, 360) * pi / 180
    return angle

def generate_trajectory(steps_amount: int, generation_logs_file: str) -> None:
    logs = {}
    try:
        state = {"x": 0, "y": 0, "angle": 0}

        for idx in range(steps_amount):

            axis = list(state.keys())[random.randint(0,2)]
            shift = {"x": 0, "y": 0, "angle": 0}

            if axis != "angle":
                # Update X or Y coord
                shift[axis] += get_random_shift()
            else:
                # Update angle
                shift[axis] += get_random_angle()

            if shift["angle"] > 2 * pi:
                shift["angle"] -= 2 * pi
            if shift["angle"] < 0:
                shift["angle"] += 2 * pi

            logs[idx] = {
                "step_idx": idx,
                "position": state,
                "shift": shift,
                "error_code": -1,
                "error_string": "",
            }

            state = {x: state[x]+shift[x] for x in state}
            logs[idx]["goal"] = state

            print(logs[idx], flush=True)

            result = walk_to_coord(
                shift["x"], shift["y"], shift["angle"]
            )

            print(f"Action result: {result}")
            logs[idx]["error_code"] = result.error_code
            logs[idx]["error_string"] = result.error_string
    except Exception as e:
        rospy.logerr(f"Exception: {e}")
        raise
    finally:
        with open(generation_logs_file, "w") as f:
            json.dump(logs, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--default-pose', default="base_nominal", help="Default pose"
    )
    parser.add_argument(
        '--default-height', default=0.215, help="Default bot height"
    )
    parser.add_argument(
        '--steps-amount', default=20, help="Amount robot's steps"
    )
    parser.add_argument(
        '--random-seed', default=1719, help="Random seed"
    )
    parser.add_argument(
        '--generation-logs-file', default="generation_logs_file.json",
        help="Path to generation log file"
    )
    parser.add_argument(
        '--trajectory-logs-file', default="trajectory_logs_file.json",
        help="Path to generation trajectory's log file"
    )
    args = parser.parse_args()

    random.seed(args.random_seed)

    rospy.init_node('change_pose')

    create_listener(args.trajectory_logs_file)

    try:
        result = set_base_into_pose(args.default_pose)
        print(result, flush=True)

        result = set_base_link_height(args.default_height)
        print(result, flush=True)

        generate_trajectory(args.steps_amount, args.generation_logs_file)

    except Exception as e:
        rospy.logerr(f"Exception: {e}")
        sys.exit(0)




if __name__ == "__main__":
    main()