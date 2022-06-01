import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from wandb.wandb_torch import torch

from data.datamodule import ImageFolderDataModule
from models.vitdino import ViTDINO


@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg["random_seed"])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 180), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomResizedCrop(size=cfg["model"]["image_size"]),
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomInvert(p=0.5),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=7),
                ],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(cfg["model"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    data = ImageFolderDataModule(
        cfg["data"]["image_folder"],
        train_image_transform=train_transform,
        val_image_transform=val_transform,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        random_seed=cfg["random_seed"],
    )
    model = ViTDINO(
        hidden_dim=cfg["model"]["hidden_dim"],
        output_dim=cfg["model"]["output_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_dim=cfg["model"]["mlp_dim"],
        image_size=cfg["model"]["image_size"],
        patch_size=cfg["model"]["patch_size"],
        projection_hidden_size=cfg["model"]["projection_hidden_size"],
        projection_layers=cfg["model"]["projection_layers"],
        learning_rate=cfg["training"]["learning_rate"],
        max_epochs=cfg["training"]["max_epochs"],
    )

    # load ckpt
    model = model.load_from_checkpoint("/home/ilyabushmakin/Documents/Projects/Playground/ProjectDINO/outputs/2022-05-30/13-32-18/models/ProjectDINO/3oo7yf7l/checkpoints/epoch=32-step=131999.ckpt")

    trainer = pl.Trainer(
        default_root_dir=cfg["training"]["checkpoints_folder"],
        gpus=cfg["training"]["gpus"],
        max_epochs=cfg["training"]["max_epochs"],
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="min",
                save_top_k=6,
                monitor="val_epoch_loss",
            ),
            LearningRateMonitor("epoch"),
        ],
        limit_train_batches=4000,
        limit_val_batches=400,
        logger=WandbLogger(project="ProjectDINO", log_model=True),
    )
    trainer.fit(model, data)
    torch.save(model.state_dict(), cfg["training"]["checkpoints_folder"] + "/last.ckpt")


if __name__ == "__main__":
    main()
