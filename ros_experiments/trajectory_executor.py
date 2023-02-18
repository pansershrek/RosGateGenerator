#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
import yaml
import time
import pickle
from functools import partial

import rospy
import actionlib

from sweetie_bot_control_msgs.msg import FollowStepSequenceActionGoal
from set_pose import set_base_into_pose, set_base_link_height

def set_new_attribute(obj, ref_obj, path = []):
    for x in ref_obj:
        if isinstance(ref_obj[x], dict):
            set_new_attribute(obj, ref_obj[x], path + [x])
        else:
            tmp = obj
            for y in path:
                tmp = getattr(tmp, y)
            setattr(tmp, x, ref_obj[x])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--default-pose', default="base_nominal", help="Default pose"
    )
    parser.add_argument(
        '--default-height', default=0.215, help="Default bot height"
    )
    parser.add_argument(
        '--trajectory-path',
        default="home/panser/Desktop/RosGateGenerator/trajectories/full_trajectory_1/trajectory_logs_file/trajectory_logs_file_1",
        help="Path to trajectory path"
    )
    args = parser.parse_args()

    pubisher = rospy.Publisher(
        '/motion/controller/step_sequence/goal',
        FollowStepSequenceActionGoal,
        queue_size=1000
    )
    rospy.init_node('change_pose')

    try:
        result = set_base_into_pose(args.default_pose)
        print(result, flush=True)

        result = set_base_link_height(args.default_height)
        print(result, flush=True)
        with open(args.trajectory_path, "rb") as f:
            trajectory = pickle.load(f)

        print("Start to send msg")
        pubisher.publish(trajectory)
        print("Sended")

    except Exception as e:
        rospy.logerr(f"Exception: {e}")
        print(traceback.format_exc())
        sys.exit(0)



if __name__ == "__main__":
    main()