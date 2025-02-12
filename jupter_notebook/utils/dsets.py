import json
import typing
from pathlib import Path
import torch
from torch.utils.data import Dataset


# This is the dataset for https://rome.baulab.info/data/dsets/known_1000.json
class KnownsDataset(Dataset):
    def __init__(self, data_path: str, *args, **kwargs):
    
        with open(data_path, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
