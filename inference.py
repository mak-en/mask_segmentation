# Inference script
from typing import Any
import warnings

import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp


from model import MyModel
from data import CovMask


def show_results(predictions: Any, display_pred: bool = True):
    images = torch.cat([x["graphics"]["image"] for x in predictions])
    masks = torch.cat([x["graphics"]["mask"] for x in predictions])
    pred_masks = torch.cat([x["graphics"]["pred_mask"] for x in predictions])
    tp = torch.cat([x["tp"] for x in predictions])
    fp = torch.cat([x["fp"] for x in predictions])
    fn = torch.cat([x["fn"] for x in predictions])
    tn = torch.cat([x["tn"] for x in predictions])
    loss = torch.stack([x["loss"] for x in predictions])

    # Calculate IoU for all the images. Needed for displaying the results.
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)

    # per image IoU means that we first calculate IoU score for each image
    # and then compute mean over these scores.
    per_image_iou_ave = smp.metrics.iou_score(
        tp, fp, fn, tn, reduction="micro-imagewise"
    )

    # dataset IoU means that we aggregate intersection and union over
    # whole dataset and then compute IoU score.
    dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

    print(
        f"AVERAGE LOSS: {torch.mean(loss)}\n"
        + f"PER_IMAGE_IOU: {per_image_iou_ave}\n"
        + f"DATASET_IOU: {dataset_iou}"
    )

    if display_pred == True:
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
        "best_models_ckpt/driven-sweep-1-epoch=02-step=116-valid_dataset_iou=0.10.ckpt"
    )

    trainer = pl.Trainer(logger=False, gpus=-1)  # gpus=-1 - use all gpus
    predictions = trainer.predict(model, dataloaders=predict_dataloader)

    show_results(predictions)
