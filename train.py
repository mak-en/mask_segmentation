import wandb
from pytorch_lightning.loggers import WandbLogger

from models import MyModel

def train():
    pass


if __name__ == "__main__":
    
    wandb.login()

    sweep_config = MyModel.get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="mask_segmentation")
    wandb.agent(sweep_id, train, count=5)