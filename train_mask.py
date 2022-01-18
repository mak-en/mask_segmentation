# Imports
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Dataset
class CovMask(Dataset):
    """
    A dataset for the segmentation of masks against covid-19.
    The link - https://www.kaggle.com/perke986/face-mask-segmentation-dataset
    """

    def __init__(self, data_path, transform=None):

        self.transform = transform

        self.data_path = data_path
        self.img_path = os.path.join(self.data_path,'data_mask/images')
        self.mask_path = os.path.join(self.data_path, 'data_mask/masks')
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)    

    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = np.array(img)

        mask = Image.open(self.mask_list[idx])
        mask = np.array(mask)

        if self.transform:
            img = self.transform(img)

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

        if self.transform:
            img = self.transform(img)

        img.show()
        mask.show()

if __name__ == '__main__':
    sd = CovMask('C:/Users/makov/Desktop')
    sd.show_example(10)