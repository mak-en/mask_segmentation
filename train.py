import os

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from model import MyModel
from data import MaskDataset


def train():
    # set up W&B logger
    wandb.init()    # required to have access to `wandb.config`
    wandb_logger = WandbLogger()

    model = MyModel("FPN", "resnet34", in_channels=3, out_classes=1)

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1,  # use all gpus
        max_epochs=2,
    )

    mask_dataset = MaskDataset(
        "C:/Users/ant_on/Desktop/",
        num_workers=os.cpu_count()
        )

    trainer.fit(
        model,
        mask_dataset
    )


if __name__ == "__main__":

    wandb.login()

    sweep_config = MyModel.get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="mask_segmentation")
    wandb.agent(sweep_id, train, count=5)
