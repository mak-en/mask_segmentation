# Imports
import os
from cv2 import transform

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import albumentations as A

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
            img = np.array(img)
            mask = np.array(mask)
            augmented = self.transform(image=img, mask=mask)
            return augmented['image'], augmented['mask']
        else:
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
        Demonstrates the 'idx' instance of the dataset.
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

    def __init__(self, data_path):
        super().__init__()
        # print(hparams)
        self.data_path = data_path
        self.batch_size = 1

        # Transforms for train subsets (different for img and mask: the mask 
        # tranforamtion does not include non affine transformations (look at
        # the target parameter in the transforamtions classes below))
        self.transform = A.Compose([A.Resize(244, 244),
                                    A.VerticalFlip(p=0.5),              
                                    A.RandomRotate90(p=0.5),
                                    A.GridDistortion(p=0.5),
                                    A.OpticalDistortion(distort_limit=2, 
                                                        shift_limit=0.5, p=1),
                                    A.CLAHE(p=0.8),
                                    A.RandomBrightnessContrast(p=0.8),    
                                    A.RandomGamma(p=0.8)])
    
    def setup(self):
        dataset = CovMask(self.data_path)
        train_size = int(0.7 * len(dataset)) # take 70% for training
        val_size = int(0.2 * len(dataset)) # take 20% for validation
        test_size = len(dataset) - (train_size + val_size) # take 10% for test

        self.train_set, self.val_set, self.test_set = \
        torch.utils.data.random_split(dataset, 
                                      [train_size, val_size, test_size])

        self.train_set.dataset.transform = self.transform
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          shuffle=False)

    def visualize_dataset(self):
        # Visualizes a piece of the train subset
        figure_img = plt.figure(figsize=(8, 8))
        figure_mask = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.train_set), size=(1,)).item()
            img, mask = self.train_set[sample_idx]
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
        _, ax_img =  plt.subplots()
        _, ax_mask =  plt.subplots()
        ax_img.imshow(img)
        ax_mask.imshow(mask)
        plt.show()

if __name__ == '__main__':
    # sd = CovMask('C:/Users/ant_on/Desktop/')
    sd = MaskDataset('C:/Users/ant_on/Desktop/')
    sd.setup()
    sd.visualize_dataloader()
    
    # a = sd.__getitem__(1)
    # to_tensor = transforms.ToTensor()
    # resize = transforms.Resize([224, 224])
    # img = resize(a[1])
    # tensor_a = to_tensor(img)
    # print(a)
    # print(tensor_a.size())

    # plt.imshow(tensor_a.permute(1, 2, 0))
    # plt.show()



