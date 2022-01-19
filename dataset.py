from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df
        self.train = train
        self.x = None
        self.y = None
        self.preprocess()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        x = torch.tensor(x).float()
        if self.train:
            y = self.y.iloc[idx]
            y = torch.tensor(y).float()
            return x, y
        else:
            return x

    def preprocess(self):
        df_dummy = self.df
        self.x = df_dummy.drop(columns=['applied'], axis=0)
        if self.train:
            self.y = df_dummy['applied']

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.train_dataset = CustomDataset(self.df, train=True)
        self.valid_dataset = CustomDataset(self.df, train=True)
        self.test_dataset = CustomDataset(self.df, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=8, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)

if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    dataset = CustomDataset(df)