import torch
import torchvision.transforms as T
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import wandb
import albumentations as A


class MyModel(pl.LightningModule):
    """Semantic Segmentation Module"""

    def __init__(
        self,
        wandb_config: dict,
        **kwargs,
    ):
        super().__init__()

        print(f"WANDB: {wandb_config}")

        # Learning rate
        self.lr = wandb_config["lr"]

        # Smp model
        self.model = smp.create_model(
            wandb_config["architecture"],
            encoder_name=wandb_config["encoder"],
            in_channels=wandb_config["in_channels"],
            classes=wandb_config["out_classes"],
            **kwargs,
        )

        # ----- Data realated stuf -----
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(wandb_config["encoder"])
        self.register_buffer(
            "std", torch.tensor(params["std"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)
        )

        # Basic data transformations needed for the model input
        self.transform = A.Resize(224, 224)

        # Transforms are model specific
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
            ]
        )
        # ------------------------------

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch):

        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height,
        # width)
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
        # but for now we just compute true positive, false positive,
        # false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        # Copy the tensors for logging
        log_image = image.clone().detach()
        log_mask = mask.clone().detach()
        log_pred_mask = pred_mask.clone().detach().int()

        # Lets's also return the image, pred_mask and mask for logging in wandb
        graphics = {
            "image": log_image,
            "mask": log_mask,
            "pred_mask": log_pred_mask,
        }

        return {
            "graphics": graphics,
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
        loss = outputs[-1]["loss"]
        graphics = outputs[-1]["graphics"]

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over
        # whole dataset
        # and then compute IoU score. The difference between dataset_iou and
        # per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap
        # could be observed.
        # Empty images influence a lot on per_image_iou and much less
        # on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Making a dict for logging in wandb
        transform = T.ToPILImage()
        mask_img = wandb.Image(
            transform(graphics["image"][0]),
            masks={
                "predictions": {
                    "mask_data": graphics["pred_mask"][0][0].cpu().numpy()
                },
                "ground_truth": {
                    "mask_data": graphics["mask"][0][0].cpu().numpy()
                },
            },
        )

        metrics = {
            # f"{stage}_graphics": mask_img,
            f"{stage}_loss": loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        wandb.log({"graphics": mask_img})
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
