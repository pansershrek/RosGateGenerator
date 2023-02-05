#!/usr/bin/env python3
import argparse
import json
import sys
import random
import yaml
from functools import partial
import traceback
from datetime import datetime

import rospy

import pickle

from set_pose import set_base_into_pose, set_base_link_height
from generate_trajectory_1 import generate_trajectory_1
from generate_trajectory_2 import generate_trajectory_2
from sweetie_bot_control_msgs.msg import FollowStepSequenceActionGoal

def callback(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)

def create_listener(trajectory_logs_file):
    rospy.Subscriber(
        "/motion/controller/step_sequence/goal",
        FollowStepSequenceActionGoal,
        partial(callback, file=trajectory_logs_file)
    )

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
        '--trajectory-amount', default=1, help="Amount robot's trajectories"
    )
    parser.add_argument(
        '--random-seed', default=datetime.now().timestamp(), help="Random seed"
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

    try:
        result = set_base_into_pose(args.default_pose)
        print(result, flush=True)

        result = set_base_link_height(args.default_height)
        print(result, flush=True)

        create_listener(args.trajectory_logs_file)
        generate_trajectory_2(args.trajectory_amount, args.generation_logs_file)

    except Exception as e:
        rospy.logerr(f"Exception: {e}")
        print(traceback.format_exc())
        sys.exit(0)




if __name__ == "__main__":
    main()