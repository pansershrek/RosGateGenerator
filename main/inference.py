#!/usr/bin/env python3

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor
from main_utils import get_final_point

def inference(model, start_point, shift, eps = 0.01):
    start_point = create_tensor_from_trajectory_point(start_point)
    cur_point = create_tensor_from_trajectory_point(start_point)
    shift = create_shift_tesor(shift)
    final_point = get_final_point(start_point, shift)
    h, c = None, None
    while torch.mean(abs(cur_point - final_point)):
        pass
