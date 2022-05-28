import os
from glob import glob
from typing import Union, Optional, Callable

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import ImageList


class ImageFolderDataModule(LightningDataModule):
    def __init__(
            self,
            image_folder: Union[str, os.PathLike],
            train_image_transform: Optional[Callable] = None,
            val_image_transform: Optional[Callable] = None,
            val_size: float = 0.01,
            test_size: float = 0.01,
            batch_size: int = 8,
            random_seed: int = 42,
            prefetch_factor: int = 4,
            num_workers: int = 4,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        image_paths = sorted(glob(os.path.join(self.image_folder, "*")))

        train_set, vt_set = train_test_split(
            image_paths,
            test_size=self.test_size + self.val_size,
            random_state=self.random_seed,
        )
        val_set, test_set = train_test_split(
            vt_set,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_seed,
        )

        self.train_dataset = ImageList(train_set, self.train_image_transform)
        self.val_dataset = ImageList(val_set, self.val_image_transform)
        self.test_dataset = ImageList(test_set, self.val_image_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
