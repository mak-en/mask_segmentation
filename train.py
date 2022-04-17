import os
import argparse
import warnings

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import yaml

from model import MyModel
from data import MaskDataset


def train():
    # required to have access to `wandb.config`
    wandb.init()
    # set up W&B logger
    wandb_logger = WandbLogger()

    print("CONFIG", wandb.config)

    trainer = pl.Trainer(
        logger=wandb_logger, gpus=-1, max_epochs=10  # gpus=-1 - use all gpus
    )

    model = MyModel(
        wandb.config.architecture,
        wandb.config.encoder,
        wandb.config.in_channels,
        wandb.config.out_classes,
        lr=wandb.config.lr,
    )

    mask_dataset = MaskDataset(
        wandb.config.data_path,
        num_workers=wandb.config.cpu_number,
        batch_size=wandb.config.batch_size,
    )

    trainer.fit(model, mask_dataset)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

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

    sweep_config["parameters"].update(
        {
            "data_path": {"value": args.data_path},
            "cpu_number": {"value": args.cpu_number},
        }
    )

    sweep_id = wandb.sweep(sweep_config, project="mask_segmentation")
    wandb.agent(sweep_id, train, count=3)
