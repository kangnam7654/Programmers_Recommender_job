import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from utils.custom_dataloader.custom_dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(self, train_df=None, valid_df=None, test_df=None):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def train_dataloader(self, train=True, batch_size=8, shuffle=True, drop_last=False):
        self.train_dataset = CustomDataset(self.train_df, train=train)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def val_dataloader(self, train=True, batch_size=8, shuffle=False, drop_last=False):
        self.valid_dataset = CustomDataset(self.valid_df, train=train)
        return DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def test_dataloader(self, train=False, batch_size=8, shuffle=True, drop_last=False):
        self.test_dataset = CustomDataset(self.test_df, train=train)
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )


if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    dataset = CustomDataset(df)
