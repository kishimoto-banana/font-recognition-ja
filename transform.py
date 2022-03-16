from abc import ABCMeta, abstractclassmethod
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import Tensor


class Transform(metaclass=ABCMeta):
    @abstractclassmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Method not implemented")


class ImageTranform:
    def __init__(self) -> None:
        self.transform = {
            "train": A.Compose(
                [
                    A.GaussNoise(var_limit=3, mean=0, p=1),
                    A.GaussianBlur(sigma_limit=[2.5, 3.5], p=1),
                    A.Normalize(mean=0, std=1, max_pixel_value=255, p=1),
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    A.Normalize(mean=0, std=1, max_pixel_value=255, p=1),
                    ToTensorV2(),
                ]
            ),
        }

    def __call__(self, img: np.array, phase: str = "train") -> Tensor:
        return self.transform[phase](image=img)["image"]
