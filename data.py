import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import albumentations as A
from typing import Optional


# Dataset
class CovMask(Dataset):
    """A dataset for the segmentation of masks against covid-19.

    The link - https://www.kaggle.com/perke986/face-mask-segmentation-dataset
    """

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, "images/")
        self.mask_path = os.path.join(self.data_path, "masks/")
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx]).convert("L")

        if self.transform:
            img = np.array(img)
            mask = np.array(mask)
            augmented = self.transform(image=img, mask=mask)
            img = np.moveaxis(augmented["image"], -1, 0)
            mask = np.array(Image.fromarray(augmented["mask"])) / 255
            mask = np.expand_dims(mask, axis=0)
            return img, mask
        else:
            return img, mask

    def get_filenames(self, path):
        """Returns a list of absolute paths to images inside given path`"""

        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

    def show_example(self, idx):
        """Demonstrates the 'idx' instance of the dataset"""

        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])

        img.show()
        mask.show()


class MaskDataset(pl.LightningDataModule):
    """Pytorch Lightning DataModule.

    Moving the transform operation here (from the CovMask(Dataset)) allows
    a flexible adjustment of the train, validate and test subsets.
    """

    def __init__(
        self,
        data_path: str,
        num_workers: Optional[int] = 1,
        batch_size: Optional[int] = 32,
    ):
        super().__init__()
        # print(hparams)
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transforms for train subsets (different for img and mask: the mask
        # tranforamtion does not include non-affine transformations (look at
        # the target parameter in the transforamtions classes below))
        self.train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
                A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
            ]
        )
        self.val_transform = A.Compose(
            [
                A.Resize(224, 224),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = CovMask(os.path.join(self.data_path, "train/"))
        self.val_dataset = CovMask(os.path.join(self.data_path, "val/"))

        self.train_dataset.transform = self.train_transform
        self.val_dataset.transform = self.val_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def visualize_dataset(self):
        # Visualizes a piece of the train subset
        figure_img = plt.figure(figsize=(8, 8))
        figure_mask = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(
                len(self.train_dataset), size=(1,)
            ).item()
            img, mask = self.train_dataset[sample_idx]
            img_ax = figure_img.add_subplot(rows, cols, i)
            mask_ax = figure_mask.add_subplot(rows, cols, i)
            img_ax.axis("off")
            mask_ax.axis("off")
            img_ax.imshow(img)
            mask_ax.imshow(mask)
        plt.show()

    def visualize_dataloader(self):
        # Displays one image and label from a train-subset batch
        train_dataloader = self.train_dataloader()
        train_imgs, train_masks = next(iter(train_dataloader))
        print(f"Images batch shape: {train_imgs.size()}")
        print(f"Masks batch shape: {train_masks.size()}")
        img = train_imgs[0]
        mask = train_masks[0]
        _, ax_img = plt.subplots()
        _, ax_mask = plt.subplots()
        ax_img.imshow(img.permute(1, 2, 0))
        ax_mask.imshow(mask.permute(1, 2, 0))
        plt.show()
