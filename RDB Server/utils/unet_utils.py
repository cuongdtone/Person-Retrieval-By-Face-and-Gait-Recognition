# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/28/2022


from typing import Tuple
import cv2
import numpy as np


def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_REFLECT_101) -> tuple:
    """Pads the image on the sides, so that it will be divisible by factor.
    Common use case: UNet type architectures.
    Args:
        image:
        factor:
        border: cv2 type border.
    Returns: padded_image
    """
    height, width = image.shape[:2]

    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(image: np.array, pads: Tuple[int, int, int, int]) -> np.ndarray:
    """Crops patch from the center so that sides are equal to pads.
    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    Returns: cropped image
    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
    height, width = image.shape[:2]

    return image[y_min_pad: height - y_max_pad, x_min_pad: width - x_max_pad]

