#!/usr/bin/env python3
import copy

import torch

from main_utils import create_tensor_from_trajectory_point, create_shift_tesor
from main_utils import get_base_position, get_final_base_position, base_is_close
from main_utils import create_full_trajectoy_point

def inference(
        model, start_point, shift, device
        eps = [0.1, 0.1, 0.1, 0.1] # x, y, w, z
    ):
    start_point = create_tensor_from_trajectory_point(start_point)[0]
    cur_point = copy.deepcopy(start_point)
    shift = create_shift_tesor(shift)
    cur_base_position = get_base_position(cur_point)
    final_base_position = get_final_base_position(cur_base_position, shift)
    h, c = None, None
    ans = []
    while not base_is_close(cur_base_position, final_base_position, eps):
        prev_point = torch.FloatTensor(
            create_full_trajectoy_point(
                shift, start_point, cur_point
            )
        )
        prev_point = prev_point.view(1, 1, -1)

        predict_point, h, c = model(prev_point.to(device), h, c)
        predict_point = predict_point.view(-1).tolist()[-35:]
        ans.append(predict_point)
        cur_base_position = get_base_position(predict_point)

    return ans
