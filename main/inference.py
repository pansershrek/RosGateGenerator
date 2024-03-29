#!/usr/bin/env python3
import copy
import math

import torch

from main_utils import create_tensor_from_trajectory_point_for_predict
from main_utils import get_base_position, get_final_base_position
from main_utils import create_shift_tesor, base_distance, is_close

def inference(
        model, start_point, shift, device,
        eps = {"x": 0.1, "y": 0.1, "angle": math.pi/6}
    ):
    model.eval()
    start_point_tensor = create_tensor_from_trajectory_point_for_predict(
        start_point
    )[0]
    cur_point = copy.deepcopy(start_point_tensor)
    cur_base_position = get_base_position(cur_point)
    final_base_position = get_final_base_position(cur_base_position, shift)

    shift = create_shift_tesor(shift)
    h, c = None, None
    ans = []

    dist = base_distance(cur_base_position, final_base_position)

    step = 0
    with open("INFERENCE_LOGS", "w") as f:
        while not is_close(dist, eps) and step < 3000:
            print(f"step: {step}, dist: {dist}", file=f)
            step += 1
            prev_point = torch.FloatTensor(
                shift + cur_point# + start_point_tensor + cur_point
            )
            prev_point = prev_point.view(1, 1, -1)
            predict_point, h, c = model(prev_point.to(device), h, c)
            predict_point = predict_point.view(-1).tolist()
            ans.append(predict_point)
            cur_base_position = get_base_position(predict_point)
            dist = base_distance(cur_base_position, final_base_position)
            cur_point = predict_point

    return ans
