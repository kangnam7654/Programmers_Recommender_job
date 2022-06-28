import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(Path(__file__).parents[2], "data")


def csv_loader(file_name):
    csv = pd.read_csv(os.path.join(DATA_DIR, file_name), encoding="utf-8")
    return csv


class CustomDataset(Dataset):
    def __init__(self, df, train=True) -> None:
        super(CustomDataset).__init__()
        self.csv = df
        self.train = train

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if self.train:
            data = torch.tensor(self.csv.iloc[idx, 3:], dtype=torch.float)
            labels = torch.tensor(self.csv.iloc[idx, 2], dtype=torch.float).unsqueeze(0)
            return data, labels
        else:
            data = self.csv.iloc[idx, 2:].to_numpy()
            return data
        
        
if __name__ == "__main__":
    train_csv = csv_loader("train_data.csv")
    test_csv = csv_loader("test_data.csv")

    dataset = CustomDataset(df=train_csv)
    for a, b in dataset:
        pass
