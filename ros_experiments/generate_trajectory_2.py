#!/usr/bin/env python3
import json
import random
from math import pi
from typing import Any

import rospy, actionlib

from sweetie_bot_clop_generator.clopper import MoveBaseGoal, MoveBaseAction

class ROSFlexbeTimeoutException(Exception):
    pass


def generate_trajectory_length(trajectory_length_restrictions: int) -> int:
    return random.randint(*trajectory_length_restrictions)

def get_x_shift(trajectory_length: int) -> list:
    shift_x_points = random.randint(1, (trajectory_length - 1) * 4)
    shift_x = shift_x_points / 10.0
    if random.random() > 0.5:
        shift_x *= -1
    return shift_x_points, shift_x

def get_y_shift(trajectory_length: int, shift_x_points: int) -> list:
    shift_y_points = 0
    if random.random() > 0.15:
        shift_y_points = random.randint(
            0, trajectory_length * 4 - shift_x_points - 1
        )
    shift_y = shift_y_points / 10.0
    if random.random() > 0.5:
        shift_y *= -1
    return shift_y_points, shift_y

def get_new_angle(
    trajectory_length: int, shift_x_points: int, shift_y_points: int
) -> float:
    new_angle = 0
    shift_angle = 0
    if (
        random.random() > 0.3 and
        (trajectory_length * 4 - shift_x_points - shift_y_points) > 0
    ):
        new_angle = random.randint(0, 360) * pi / 180
        shift_angle = 1
    if random.random() > 0.5:
        new_angle *= -1
    return shift_angle, new_angle

def walk_to_coord(new_state: dict, trajectory_length: int) -> Any:
    msg = MoveBaseGoal(
        gait_type = "walk_overlap",
        n_steps = trajectory_length * 4,
        duration = trajectory_length * 3.4
    )
    msg.setTargetBaseShift(
        x = new_state["x"],
        y = new_state["y"],
        angle = new_state["angle"]
    )
    msg.addEndEffectorsTargets(["leg1","leg2","leg3","leg4"])

    client = actionlib.SimpleActionClient(
        'clop_generator', MoveBaseAction
    )
    if not client.wait_for_server(
        timeout=rospy.Duration(trajectory_length * 3.0)
    ):
        raise ROSFlexbeTimeoutException(
            'clop_generator action is not available.'
        )

    client.send_goal(msg)
    client.wait_for_result()

    return client.get_result()

def generate_trajectory_2(
    trajectory_amount: int, generation_logs_file: str,
    trajectory_length_restrictions: list = [1, 10]
) -> None:
    logs = {}
    try:
        state = {"x": 0, "y": 0, "angle": 0}

        for idx in range(trajectory_amount):
            trajectory_length = generate_trajectory_length(
                trajectory_length_restrictions
            )
            shift_x_points, shift_x = get_x_shift(trajectory_length)
            shift_y_points, shift_y = get_y_shift(
                trajectory_length, shift_x_points
            )
            shift_angle_points, new_angle = get_new_angle(
                trajectory_length, shift_x_points, shift_y_points
            )
            shift = {"x": shift_x, "y": shift_y, "angle": new_angle}

            new_state = {}
            for x in state:
                new_state[x] = state[x] + shift[x]

            if new_state["angle"] > 2 * pi:
                new_state["angle"] -= 2 * pi
            if new_state["angle"] < 0:
                new_state["angle"] += 2 * pi

            logs[idx] = {
                "trajectory_idx": idx,
                "state": state,
                "new_state": new_state,
                "shift": shift,
                "error_code": -1,
                "error_string": "",
            }
            print(f"State: {state}, new_state: {new_state}")
            result = walk_to_coord(
                shift, shift_x_points + shift_y_points + shift_angle_points
            )
            print(f"Result: {result}")
            logs[idx]["error_code"] = result.error_code
            logs[idx]["error_string"] = result.error_string
    except Exception as e:
        rospy.logerr(f"Exception: {e}")
        raise
    finally:
        with open(generation_logs_file, "w") as f:
            json.dump(logs, f)
