#!/usr/bin/env python3
import rospy, actionlib

from flexbe_msgs.msg import BehaviorExecutionAction, BehaviorExecutionGoal


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