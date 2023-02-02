#!/usr/bin/env python3
from math import pi
import json
import random

import rospy, actionlib

from sweetie_bot_clop_generator.clopper import MoveBaseGoal, MoveBaseAction

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

def generate_trajectory_1(steps_amount: int, generation_logs_file: str) -> None:
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
