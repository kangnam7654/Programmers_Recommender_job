from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df
        self.train = train
        self.x, self.y = self.preprocess()


    def __len__(self):
        len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def preprocess(self):
        df_dummy = pd.get_dummies(self.df)
        x = df_dummy.drop(columns=['applied'], axis=0)
        if self.train:
            y = df_dummy['applied']
            return x, y
        else:
            return x

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.train_dataset = CustomDataset(df, train=True)
        self.valid_dataset = CustomDataset(df, train=True)
        self.test_dataset = CustomDataset(df, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    dataset = CustomDataset(df)