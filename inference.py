# Inference script
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import MyModel
from data import CovMask

if __name__ == "__main__":
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

    trainer = pl.Trainer(gpus=-1)  # gpus=-1 - use all gpus
    predictions = trainer.predict(model, dataloaders=predict_dataloader)
