"""Image and mask transforms for training and evaluation."""

from __future__ import annotations

import cv2
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from typing import Any


class SegmentationTransform:
    def __init__(self, base_config: dict[str, Any], train: bool):
        self._config: dict = base_config["transforms"]
        image_size = base_config["dataset"]["input_size"]
        self._new_width = image_size[0]
        self._new_height = image_size[1]
        self._train = train

        self._train_transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(self._new_height, self._new_width)),
                A.PadIfNeeded(
                    self._new_height,
                    self._new_width,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.HorizontalFlip(p=self._config["prob_horizontal_flip"]),
                A.RandomBrightnessContrast(p=self._config["prob_brightness_contrast"]),
                A.ShiftScaleRotate(
                    shift_limit=self._config["shift_limit"],
                    scale_limit=self._config["scale_limit"],
                    rotate_limit=self._config["rotate_limit"],
                    border_mode=cv2.BORDER_CONSTANT,
                    p=self._config["prob_random_shift_scale_rotate"],
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self._eval_transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(self._new_height, self._new_width)),
                A.PadIfNeeded(
                    self._new_height,
                    self._new_width,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def apply(self, image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self._train:
            result = self._train_transform(image=image, mask=mask)
        else:
            result = self._eval_transform(image=image, mask=mask)
        return result["image"], result["mask"].to(torch.long)
