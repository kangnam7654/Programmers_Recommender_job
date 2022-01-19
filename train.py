import pytorch_lightning as pl
from model import Model, Backbone
from dataset import CustomDataModule
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

def main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    df = pd.read_csv('./data/train.csv')
    df = pd.get_dummies(df)
    train_df, valid_df = train_test_split(df, train_size=0.8)
    train_loader = CustomDataModule(df=train_df).train_dataloader()
    valid_loader = CustomDataModule(df=valid_df).val_dataloader()

    # ------------
    # model
    # ------------
    model = Model(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, gpus=1)
    trainer.fit(model, train_loader, valid_loader)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    main()