import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = os.path.join(ROOT_DIR, "data")

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from utils.custom_dataloader.custom_dataloader import CustomDataModule
from model.model import BinaryClassification


def main():
    pl.seed_everything(1234)
    # args
    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = Model.add_model_specific_args(parser)
    # args = parser.parse_args()

    # data
    train_csv = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

    train_df, valid_df = train_test_split(
        train_csv, train_size=0.8, stratify=train_csv["applied"]
    )

    # data module
    data_module = CustomDataModule(
        train_df=train_df, valid_df=valid_df, test_df=test_df
    )
    train_loader = data_module.train_dataloader(batch_size=256)
    valid_loader = data_module.val_dataloader(batch_size=10000)
    test_loader = data_module.predict_dataloader(batch_size=10000)

    # early stopping
    early_stop = EarlyStopping(
        monitor="valid_acc", patience=10, verbose=True, mode="max"
    )
    checkpoint = ModelCheckpoint(monitor="valid_acc", save_last=False, mode='max')

    # model
    model = BinaryClassification()
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        callbacks=[early_stop, checkpoint],
        gpus=1,
        num_sanity_val_steps=0,
    )

    # training
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


def inference():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

    data_module = CustomDataModule(test_df=test_df)
    test_loader = data_module.predict_dataloader(batch_size=10000)

    model = BinaryClassification()
    model = model.load_from_checkpoint(
        "D:\project\programmers\job_recommendation\lightning_logs\\version_38\checkpoints\epoch=22-step=874.ckpt"
    )
    trainer = pl.Trainer(
        max_epochs=1000, accelerator="gpu", gpus=1, num_sanity_val_steps=0
    )

    # training
    cls = trainer.predict(model=model, dataloaders=test_loader)
    all_cls = torch.cat(cls).detach().cpu().numpy()
    submit = pd.DataFrame(columns=["applied"], data=all_cls)
    submit.to_csv("submit.csv", index=False)
    pass


if __name__ == "__main__":
    main()
    # inference()
