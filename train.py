import os
import argparse

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import yaml
from typing import Optional

from model import MyModel
from data import MaskDataset


def train(data_path: str, cpu_number: Optional[int] = 1):
    # required to have access to `wandb.config`
    wandb.init()
    # set up W&B logger
    wandb_logger = WandbLogger()

    model = MyModel(
        wandb.config.architecture,
        wandb.config.encoder,
        wandb.config.in_channels,
        wandb.config.out_classes,
        lr=wandb.config.lr
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1,  # use all gpus
    )

    mask_dataset = MaskDataset(
        data_path, num_workers=cpu_number, batch_size=wandb.config.batch_size
    )

    trainer.fit(model, mask_dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mask segmentation")
    parser.add_argument("--data_path", type=str)
    parser.add_argument(
        "--cpu_number", default=os.cpu_count(), type=int, help="number of cpus"
    )
    args = parser.parse_args()

    wandb.login()

    # Create a dict bsed on the yaml config file
    with open("fpn_resnet34_config.yaml") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, project="mask_segmentation")
    wandb.agent(sweep_id, train(args.data_path, args.cpu_number), count=5)
