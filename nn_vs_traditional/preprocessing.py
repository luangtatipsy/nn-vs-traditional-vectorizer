from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from typing_extensions import Self


class ImagePreprocessor:
    def __init__(self, image_size: Tuple[int, int]) -> None:
        self.width, self.height = image_size

    def pad(self, img: np.ndarray) -> np.ndarray:
        img_height, img_width, _ = img.shape

        height_padding = img_width - img_height if img_width > img_height else 0
        width_padding = img_height - img_width if img_height > img_width else 0

        top_padding = height_padding // 2
        bottom_padding = top_padding if height_padding % 2 == 0 else top_padding + 1

        left_padding = width_padding // 2
        right_padding = left_padding if width_padding % 2 == 0 else left_padding + 1

        padding = [(top_padding, bottom_padding), (left_padding, right_padding), (0, 0)]

        return np.pad(array=img, pad_width=padding, mode="constant", constant_values=0)

    def resize(self, img: np.ndarray, desired_dim: Tuple[int, int]):
        return cv2.resize(img, desired_dim, interpolation=cv2.INTER_CUBIC)

    def fit(self, X, y=None, **fit_params) -> Self:
        return self

    def transform(self, X: List[Image.Image]) -> np.ndarray:
        imgs = [self.pad(np.array(img)) for img in X]

        return np.array(
            [self.resize(img, desired_dim=(self.width, self.height)) for img in imgs]
        )
