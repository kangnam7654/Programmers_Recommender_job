import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')

import pandas as pd
import pytorch_lightning as pl
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
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))

    # train_df, valid_df = train_test_split(train_csv, train_size=0.8, stratify=train_csv['applied'])

    # data module
    data_module = CustomDataModule(train_df=train_df, test_df=test_df)
    train_loader = data_module.train_dataloader(batch_size=6000)
    # valid_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # model
    model = BinaryClassification()
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', gpus=1, num_sanity_val_steps=0)

    # training    
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()