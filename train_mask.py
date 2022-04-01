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
import segmentation_models_pytorch as smp
import wandb
from pytorch_lightning.loggers import WandbLogger

# Sweep initial parameters
hyperparameter_defaults = dict(
    data_path='data_semantics',
    batch_size=2,
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
        mask = Image.open(self.mask_list[idx]).convert("L")

        if self.transform:
            img = np.array(img)
            mask = np.array(mask)
            augmented = self.transform(image=img, mask=mask)
            img = np.moveaxis(augmented['image'], -1, 0)
            mask = np.array(Image.fromarray(augmented['mask'])) / 255
            mask = np.expand_dims(mask, axis=0)
            return img, mask
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
        self.batch_size = 16

        # Transforms for train subsets (different for img and mask: the mask 
        # tranforamtion does not include non affine transformations (look at
        # the target parameter in the transforamtions classes below))
        self.transform = A.Compose([A.Resize(224, 224),
                                    A.VerticalFlip(p=0.5),              
                                    A.RandomRotate90(p=0.5),
                                    A.GridDistortion(p=0.5),
                                    A.OpticalDistortion(distort_limit=2, 
                                                        shift_limit=0.5, p=1),
                                    A.CLAHE(p=0.8),
                                    A.RandomBrightnessContrast(p=0.8),    
                                    A.RandomGamma(p=0.8)])
    
    def setup(self, stage=None):
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
                          shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=2)

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
        ax_img.imshow(img.permute(1, 2, 0))
        ax_mask.imshow(mask.permute(1, 2, 0))
        plt.show()

class MyModel(pl.LightningModule):
    '''
    Semantic Segmentation Module
    '''

    def __init__(
        self, arch, encoder_name, in_channels, out_classes, **kwargs
        ):
        super().__init__()

        # self.lr = hparams.lr
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, 
            classes=out_classes, 
            **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have 
        # [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually 
        # encoder have 5 stages of downsampling by factor 2 (2 ^ 5 = 32);
        #  e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 
        # 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1

        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for 
        # binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param 
        # `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
            )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def repeat():
    # set up W&B logger
    wandb.init()    # required to have access to `wandb.config`
    wandb_logger = WandbLogger()

    model = MyModel("FPN", "resnet34", in_channels=3, out_classes=1)

    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=2,
    )

    mask_dataset = MaskDataset("C:/Users/ant_on/Desktop/")

    trainer.fit(
        model, 
        mask_dataset
    )

if __name__ == '__main__':
    pass

#    sd = MaskDataset("C:/Users/ant_on/Desktop/")
#    sd.setup()
#    sd.visualize_dataloader()
    # sweep_id = wandb.sweep(sweep_config, project="Mask segmentation")

    # wandb.agent(sweep_id, function=repeat)

    # # run validation dataset
    # valid_metrics = trainer.validate(
    #     model, dataloaders=mask_dataset.val_dataloader(), verbose=False
    #     )
    # print(valid_metrics)

    # # run test dataset
    # test_metrics = trainer.test(
    #     model, dataloaders=mask_dataset.test_dataloader(), verbose=False
    #     )
    # print(test_metrics)

    # batch = next(iter(mask_dataset.test_dataloader()))
    # with torch.no_grad():
    #     model.eval()
    #     logits = model(batch[0])
    # pr_masks = logits.sigmoid()

    # for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    #     plt.figure(figsize=(10, 5))

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    #     plt.title("Image")
    #     plt.axis("off")

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, 
    #                                           # because we have only one class
    #     plt.title("Ground truth")
    #     plt.axis("off")

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, 
    #                                           # because we have only one class
    #     plt.title("Prediction")
    #     plt.axis("off")

    #     plt.show()

