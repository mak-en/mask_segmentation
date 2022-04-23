# Inference script
from typing import Any
import warnings

import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import MyModel
from data import CovMask


def show_results(predictions: Any, display_pred: bool = True):
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Transform
    predict_transform = A.Compose([A.Resize(224, 224)])

    # Data
    predict_dataset = CovMask("data/test/", transform=predict_transform)
    predict_dataloader = DataLoader(
        predict_dataset, batch_size=4, shuffle=False, num_workers=1
    )

    # Model
    model = MyModel.load_from_checkpoint(
        "best_models_ckpt/elated-sweep-1-epoch=01-step=019-valid_dataset_iou=0.04.ckpt"
    )

    trainer = pl.Trainer(logger=False, gpus=-1)  # gpus=-1 - use all gpus
    predictions = trainer.predict(model, dataloaders=predict_dataloader)

    show_results(predictions)
