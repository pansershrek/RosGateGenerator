#!/usr/bin/env python3

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor

def inference(model, start_point, shift):
    start_point = create_tensor_from_trajectory_point(start_point)
    shift = create_shift_tesor(shift)
    pass
