from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class CustomDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df
        self.train = train
        self._preprocess()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        x = torch.tensor(x)
        if self.train:
            y = self.y.iloc[idx]
            y = torch.tensor(y)
            return x, y
        else:
            return x

    def _preprocess(self):
        if self.train:
            self.x = self.df.drop(columns=['applied'], axis=0)
            self.y = self.df['applied']
        else:
            self.x = self.df

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def train_dataloader(self):
        self.train_dataset = CustomDataset(self.df, train=True)
        return DataLoader(self.train_dataset, batch_size=8, shuffle=True, drop_last=False)

    def val_dataloader(self):
        self.valid_dataset = CustomDataset(self.df, train=True)
        return DataLoader(self.valid_dataset, batch_size=8, shuffle=False, drop_last=False)

    def test_dataloader(self):
        self.test_dataset = CustomDataset(self.df, train=False)
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=False)

if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    dataset = CustomDataset(df)