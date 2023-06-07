#!/usr/bin/env python3
import copy
import math

import torch

from main_utils import create_tensor_from_trajectory_point_for_predict
from main_utils import get_base_position, get_final_base_position
from main_utils import create_shift_tesor, base_distance, is_close

def inference_ff(
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
    ans = [start_point_tensor, ]

    prev_point = torch.FloatTensor(
        shift + cur_point# + start_point_tensor + cur_point
    )
    prev_point = prev_point.view(1, -1)
    predict_points = model(prev_point.to(device))
    predict_points = predict_points[0].tolist()
    for x in range(0, 19):
        ans.append(predict_points[x * 35: (x+1) * 35])
    prev_point = torch.FloatTensor(
        shift + ans[-1]# + start_point_tensor + cur_point
    )
    prev_point = prev_point.view(1, -1)
    predict_points = model(prev_point.to(device))
    predict_points = predict_points[0].tolist()
    #for x in range(0, 19):
    for x in range(0, 19):
        ans.append(predict_points[x * 35: (x+1) * 35])
    prev_point = torch.FloatTensor(
        shift + ans[-1]# + start_point_tensor + cur_point
    )
    prev_point = prev_point.view(1, -1)
    predict_points = model(prev_point.to(device))
    predict_points = predict_points[0].tolist()
    #for x in range(0, 19):
    for x in range(0, 19):
        ans.append(predict_points[x * 35: (x+1) * 35])
    prev_point = torch.FloatTensor(
        shift + ans[-1]# + start_point_tensor + cur_point
    )
    prev_point = prev_point.view(1, -1)
    predict_points = model(prev_point.to(device))
    predict_points = predict_points[0].tolist()
    #for x in range(0, 19):
    for x in range(0, 19):
        ans.append(predict_points[x * 35: (x+1) * 35])

    #prev_point = torch.FloatTensor(
    #    shift + ans[-1]# + start_point_tensor + cur_point
    #)
    #prev_point = prev_point.view(1, -1)
    #predict_points = model(prev_point.to(device))
    #predict_points = predict_points[0].tolist()
    #for x in range(0, 19):
    #    ans.append(predict_points[x * 35: (x+1) * 35])


    return ans
