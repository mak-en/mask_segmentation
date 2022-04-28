# Inference script
from typing import Any
import warnings
import argparse

import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


from model import MyModel
from data import CovMask


def show_results(
    predictions: Any, display_pred: bool = True, batch_size: int = 1
):
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

    # Depicts the results of the test from the last batch.
    # If the batch size is less then 4, shows only 1 image, ohterwise 3.
    if display_pred is True:
        if batch_size < 4:
            figure_img = plt.figure(figsize=(24, 8))
            cols, rows = 3, 1
            graphics = {}
            graphics["images"] = images[-1]
            graphics["masks"] = masks[-1]
            graphics["pred_masks"] = pred_masks[-1]
            idx = 0
            for j in range(cols):
                if j == 0:
                    idx += 1
                    fig_ax = figure_img.add_subplot(rows, cols, idx)
                    fig_ax.axis("off")
                    fig_ax.imshow(graphics["images"].permute(1, 2, 0))
                elif j == 1:
                    idx += 1
                    fig_ax = figure_img.add_subplot(rows, cols, idx)
                    fig_ax.axis("off")
                    fig_ax.imshow(graphics["masks"].permute(1, 2, 0))
                else:
                    idx += 1
                    fig_ax = figure_img.add_subplot(rows, cols, idx)
                    fig_ax.axis("off")
                    fig_ax.title.set_text(f"IoU: {per_image_iou[-1].item()}")
                    fig_ax.imshow(graphics["pred_masks"].permute(1, 2, 0))
            plt.show()
        else:
            figure_img = plt.figure(figsize=(24, 12))
            cols, rows = 3, 3
            graphics = {}
            graphics["images"] = images[-3:]
            graphics["masks"] = masks[-3:]
            graphics["pred_masks"] = pred_masks[-3:]
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if j == 0:
                        idx += 1
                        fig_ax = figure_img.add_subplot(rows, cols, idx)
                        fig_ax.axis("off")
                        fig_ax.imshow(graphics["images"][i].permute(1, 2, 0))
                    elif j == 1:
                        idx += 1
                        fig_ax = figure_img.add_subplot(rows, cols, idx)
                        fig_ax.axis("off")
                        fig_ax.imshow(graphics["masks"][i].permute(1, 2, 0))
                    else:
                        idx += 1
                        fig_ax = figure_img.add_subplot(rows, cols, idx)
                        fig_ax.axis("off")
                        fig_ax.title.set_text(
                            f"IoU: {per_image_iou[-3+i].item()}"
                        )
                        fig_ax.imshow(
                            graphics["pred_masks"][i].permute(1, 2, 0)
                        )
            plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Parser for the script arguments
    parser = argparse.ArgumentParser(description="Inference mask segmentation")
    parser.add_argument(
        "--data_test_path", type=str, help="path to the test data folder"
    )
    parser.add_argument(
        "--best_model_path", type=str, help="path to the best model"
    )
    args = parser.parse_args()

    # Transform
    predict_transform = A.Compose([A.Resize(224, 224)])

    # Data
    predict_dataset = CovMask(args.data_test_path, transform=predict_transform)
    predict_dataloader = DataLoader(
        predict_dataset, batch_size=4, shuffle=False, num_workers=1
    )

    # Model
    model = MyModel.load_from_checkpoint(args.best_model_path)

    # Pytorch lightning trainer
    trainer = pl.Trainer(logger=False, gpus=-1)  # gpus=-1 - use all gpus
    predictions = trainer.predict(model, dataloaders=predict_dataloader)

    show_results(predictions, batch_size=4)

