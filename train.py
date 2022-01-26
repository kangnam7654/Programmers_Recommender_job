import pytorch_lightning as pl
from model import Model, Backbone
from dataset import CustomDataModule
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from EDA import EDA

def main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------
    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = Model.add_model_specific_args(parser)
    # args = parser.parse_args()

    # ------------
    # data
    # ------------
    eda = EDA()
    train_csv = eda.train_csv
    test_csv = eda.test_csv

    train_df, valid_df = train_test_split(train_csv, train_size=0.8, stratify=train_csv['applied'])
    train_loader = CustomDataModule(df=train_df).train_dataloader()
    valid_loader = CustomDataModule(df=valid_df).val_dataloader()
    test_loader = CustomDataModule(df=test_csv).test_dataloader()

    # ------------
    # model
    # ------------
    model = Model(Backbone(hidden_dim=2), learning_rate=1e-3)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(model, train_loader, valid_loader)

    # ------------
    # testing
    # ------------
    result = trainer.predict(model, test_loader)
    print(result)


if __name__ == '__main__':
    main()