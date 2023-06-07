#!/usr/bin/env python3
import copy
import math

import torch

from main_utils import create_tensor_from_trajectory_point_for_predict
from main_utils import get_base_position, get_final_base_position
from main_utils import create_shift_tesor, base_distance, is_close

def inference_sep(
        model_head, model_legs1, model_legs2, model_legs3, model_legs4, start_point, shift, device,
        eps = {"x": 0.1, "y": 0.1, "angle": math.pi/6}
    ):

    model_head.eval()
    model_legs1.eval()
    model_legs2.eval()
    model_legs3.eval()
    model_legs4.eval()
    start_point_tensor = create_tensor_from_trajectory_point_for_predict(
        start_point
    )[0]
    cur_point = copy.deepcopy(start_point_tensor)
    cur_base_position = get_base_position(cur_point)

    shift = create_shift_tesor(shift)
    ans = [start_point_tensor, ]

    prev_point = torch.FloatTensor(
        shift + cur_point[:7]
    )
    prev_point1 = torch.FloatTensor(
        shift + cur_point[7:14]
    )

    prev_point2 = torch.FloatTensor(
        shift + cur_point[14:21]
    )
    prev_point3 = torch.FloatTensor(
        shift + cur_point[21:28]
    )
    prev_point4 = torch.FloatTensor(
        shift + cur_point[28:]
    )
    prev_point = prev_point.view(1, -1)

    prev_point1 = prev_point1.view(1, -1)
    prev_point2 = prev_point2.view(1, -1)
    prev_point3 = prev_point3.view(1, -1)
    prev_point4 = prev_point4.view(1, -1)
    predict_points_head = model_head(prev_point.to(device))
    predict_points_head = predict_points_head[0].tolist()
    predict_points_legs1 = model_legs1(prev_point1.to(device))
    predict_points_legs1 = predict_points_legs1[0].tolist()
    predict_points_legs2 = model_legs2(prev_point2.to(device))
    predict_points_legs2 = predict_points_legs2[0].tolist()
    predict_points_legs3 = model_legs3(prev_point3.to(device))
    predict_points_legs3 = predict_points_legs3[0].tolist()
    predict_points_legs4 = model_legs4(prev_point4.to(device))
    predict_points_legs4 = predict_points_legs4[0].tolist()
    for x in range(0, 19):
        tmp =(
            predict_points_head[x * 7: (x+1) * 7] +
            predict_points_legs1[x * 7: (x+1) * 7] +
            predict_points_legs2[x * 7: (x+1) * 7] +
            predict_points_legs3[x * 7: (x+1) * 7] +
            predict_points_legs4[x * 7: (x+1) * 7]
        )
        ans.append(tmp)
    cur_point = copy.deepcopy(ans[-1])
    prev_point = torch.FloatTensor(
        shift + cur_point[:7]
    )
    prev_point1 = torch.FloatTensor(
        shift + cur_point[7:14]
    )
    prev_point2 = torch.FloatTensor(
        shift + cur_point[14:21]
    )
    prev_point3 = torch.FloatTensor(
        shift + cur_point[21:28]
    )
    prev_point4 = torch.FloatTensor(
        shift + cur_point[28:]
    )
    prev_point = prev_point.view(1, -1)
    prev_point1 = prev_point1.view(1, -1)
    prev_point2 = prev_point2.view(1, -1)
    prev_point3 = prev_point3.view(1, -1)
    prev_point4 = prev_point4.view(1, -1)
    predict_points_head = model_head(prev_point.to(device))
    predict_points_head = predict_points_head[0].tolist()
    predict_points_legs1 = model_legs1(prev_point1.to(device))
    predict_points_legs1 = predict_points_legs1[0].tolist()
    predict_points_legs2 = model_legs2(prev_point2.to(device))
    predict_points_legs2 = predict_points_legs2[0].tolist()
    predict_points_legs3 = model_legs3(prev_point3.to(device))
    predict_points_legs3 = predict_points_legs3[0].tolist()
    predict_points_legs4 = model_legs4(prev_point4.to(device))
    predict_points_legs4 = predict_points_legs4[0].tolist()
    for x in range(0, 19):
        tmp =(
            predict_points_head[x * 7: (x+1) * 7] +
            predict_points_legs1[x * 7: (x+1) * 7] +
            predict_points_legs2[x * 7: (x+1) * 7] +
            predict_points_legs3[x * 7: (x+1) * 7] +
            predict_points_legs4[x * 7: (x+1) * 7]
        )
        ans.append(tmp)
    return ans
