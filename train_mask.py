# Imports
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# Sweep initial parameters
hyperparameter_defaults = dict(
    data_path='data_semantics',
    batch_size = 2,
    lr = 1e-3,
    num_layers = 5,
    features_start = 64,
    bilinear = False,
    grad_batches = 1,
    epochs = 20
)

# Dataset
class CovMask(Dataset):
    """
    A dataset for the segmentation of masks against covid-19.
    The link - https://www.kaggle.com/perke986/face-mask-segmentation-dataset
    """

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.img_path = os.path.join(self.data_path,'data_mask/images')
        self.mask_path = os.path.join(self.data_path, 'data_mask/masks')
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        self.transform = transform    

    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        # img = np.array(img)
        # mask = np.array(mask)

        return img, mask

    def get_filenames(self, path):
        '''
        Returns a list of absolute paths to images inside given `path`
        '''
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list
    
    def show_example(self, idx):
        '''
        Demonstrates an 'idx' instance of the dataset.
        If there is self.transformation, shows with the applied transformation
        '''
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])

        img.show()
        mask.show()

class MaskDataset(pl.LightningDataModule):
    '''
    Pytorch Lightning DataModule.
    Moving the transform operation here (from the CovMask(Dataset)) allows
    a flexible adjustment of the train, validate and test subsets.
    '''

    def __init__(self, data_path, transform=None):
        super().__init__()
        # print(hparams)
        self.data_path = data_path
        self.batch_size = 1

        # Transforms for train, val, test subsets (can be different)
        if transform:
            self.train_transform = transform
            self.val_transform = transform
            self.test_transform = transform
        else:
            transform = transforms.Compose([
                transforms.ColorJitter(hue=.20, saturation=.20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                     std=[0.32064945, 0.32098866, 0.32325324])
            ])
            self.train_transform = transform
            self.val_transform = transform
            self.test_transform = transform
    
    def setup(self):
        dataset = CovMask(self.data_path)
        train_size = int(0.7 * len(dataset)) # take 70% for training
        val_size = int(0.2 * len(dataset)) # take 20% for validation
        test_size = len(dataset) - (train_size + val_size) # take 10% for test

        self.train_set, self.val_set, self.test_set = \
        torch.utils.data.random_split(dataset, 
                                      [train_size, val_size, test_size])

        self.train_set.dataset.transform = self.train_transform
        self.val_set.dataset.transform = self.val_transform
        self.test_set.dataset.transform = self.test_transform
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          shuffle=False)

    def visualize_dataset(self):
        # Visualizes a piece of the train subset
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.train_set), size=(1,)).item()
            norm_img, mask = self.train_set[sample_idx]
            # mean = torch.tensor([0.485, 0.456, 0.406])
            # std = torch.tensor([0.229, 0.224, 0.225])
            # img = norm_img * std[:, None, None] + mean[:, None, None] 
            figure.add_subplot(rows, cols, i)
            # plt.title(self.labels_map[label])
            plt.axis("off")
            plt.imshow(norm_img.permute(1, 2, 0))
            # plt.imshow(norm_img)
        plt.show()

    def visualize_dataloader(self):
        # Display one image and label from a train-subset batch
        train_dataloader = self.train_dataloader()
        train_features, train_labels = next(iter(train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        norm_img = train_features[0]
        # mean = torch.tensor([0.485, 0.456, 0.406])
        # std = torch.tensor([0.229, 0.224, 0.225])
        # img = norm_img * std[:, None, None] + mean[:, None, None]
        mask = train_labels[0]
        plt.imshow(norm_img.permute(1, 2, 0))
        # plt.imshow(norm_img)
        plt.show()
        # print(f"Label: {self.labels_map[label.item()]}")

if __name__ == '__main__':
    sd = MaskDataset('C:/Users/makov/Desktop')
    sd.setup()
    sd.visualize_dataloader()