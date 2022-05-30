import torch
from pytorch_lightning import LightningModule
from torch import optim
from vit_pytorch import ViT
from vit_pytorch.dino import Dino

from models.utils import linear_annealing_with_plateau


class ViTDINO(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_dim: int = 512,
        image_size: int = 256,
        patch_size: int = 32,
        projection_hidden_size: int = 256,
        projection_layers: int = 3,
        learning_rate: float = 0.001,
        max_epochs: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=output_dim,
            dim=hidden_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=0.15
        )
        self.learner = Dino(
            self.model,
            image_size=image_size,
            hidden_layer="to_latent",
            projection_hidden_size=projection_hidden_size,  # projector network hidden dimension
            projection_layers=projection_layers,  # number of layers in projection network
            num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
            student_temp=0.9,  # student temperature
            teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
            global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay=0.95,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay=0.95,
            # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        )

        self.teacher_temp_scheduler = linear_annealing_with_plateau(0.04, 0.07, 4000*5)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.learner.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=16,
            verbose=False,
            cooldown=32,
            min_lr=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }

    def training_step(self, batch, batch_idx):
        self.learner.update_moving_average()
        self.learner.teacher_temp = next(self.teacher_temp_scheduler)
        loss = self.learner(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.learner(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.learner(batch)
        self.log("test_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        loss = (
            torch.cat([output["loss"].reshape(1) for output in outputs])
            .mean()
            .cpu()
            .numpy()
        )

        self.log("train_epoch_loss", loss.item())

    def validation_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        loss = torch.cat([output.reshape(1) for output in outputs]).mean().cpu().numpy()

        self.log("val_epoch_loss", loss.item())

    def test_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        loss = torch.cat([output.reshape(1) for output in outputs]).mean().cpu().numpy()

        self.log("test_epoch_loss", loss.item())
