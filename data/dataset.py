import os
from typing import Callable, Optional, Collection, Union

from PIL import Image
from torch.utils.data import Dataset

from data.utils import drop_alpha


class ImageList(Dataset):
    def __init__(
            self,
            images: Collection[Union[str, os.PathLike]],
            transform: Optional[Callable] = None,
    ):
        self.images = sorted(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        image = drop_alpha(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


