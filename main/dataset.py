import os
#!/usr/bin/env python3
import json
import pickle
import yaml

from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, trajectories_path: str, generations_path: str) -> None:
        self.idx2data = {}
        for x in os.listdir(trajectories_path):
            self.idx2data[len(self.idx2data)] = {
                "trajectory": os.path.join(trajectories_path, x),
                "generation": os.path.join(
                    generations_path,
                    f"generation_logs_file_{x.split('_')[-1]}"
                )
            }

    def __len__(self) -> int:
        return len(self.idx2data)

    def __getitem__(self, idx: int):
        with open(self.idx2data[idx]["trajectory"], "rb") as f:
            trajectory = pickle.load(f)
            trajectory = yaml.safe_load(str(trajectory))
        with open(self.idx2data[idx]["generation"], "r") as f:
            generation = json.load(f)
        if generation["0"]["error_code"] != 0:
            random_item = random.randint(0, len(self.idx2data) - 1)
            return self.__getitem__(random_item)
        data = {
            "shift": generation["0"]["shift"],
            "points": []
        }
        for x in trajectory["goal"]["base_motion"]["points"]:
            data["points"].append(
                {
                    "step": len(data["points"]),
                    "base_motion": x
                }
            )
        for leg_position in trajectory["goal"]["ee_motion"]:
            for idx, x in enumerate(leg_position["points"]):
                data["points"][idx][leg_position["name"]] = x
        return data