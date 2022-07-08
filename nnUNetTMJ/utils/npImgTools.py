# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: npImgTools.py                                          | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import numpy as np
from typing import Union, List


def stretchArr(
    arr: np.ndarray, min_val: Union[int, float] = 0, max_val: Union[int, float] = 255
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise Exception("Input should be an ndarray")
    arr = arr.astype(float)
    a = (arr - arr.min()) / (arr.max() - arr.min())
    a = a * (max_val - min_val) + min_val
    return a


def mapMatUint8(arr: np.ndarray) -> np.ndarray:
    return stretchArr(arr, 0, 255).astype(np.uint8)


def imgChannel(img: np.ndarray) -> int:
    if len(img.shape) == 3:
        return img.shape[2]
    if len(img.shape) == 2:
        return 1


def gray2rgb(img: np.ndarray) -> np.ndarray:
    new_img = np.concatenate(
        (img[:, :, np.newaxis], img[:, :, np.newaxis], img[:, :, np.newaxis]), axis=2
    )
    return new_img


def overlapMask(
    img: np.ndarray, mask: np.ndarray, color: Union[tuple, list] = (255, 0, 0), alpha=1
) -> np.ndarray:
    if imgChannel(img) == 1:
        img = gray2rgb(img)
    if imgChannel(mask) == 1:
        mask = gray2rgb(mask)
    im = img.astype(float)
    channel = np.ones(img.shape[:2], np.float)
    color_ = np.concatenate(
        (
            channel[:, :, np.newaxis] * color[0],
            channel[:, :, np.newaxis] * color[1],
            channel[:, :, np.newaxis] * color[2],
        ),
        axis=2,
    )
    f_im = im * (1 - mask) + im * mask * (1 - alpha) + color_ * alpha * mask
    return f_im.astype(np.uint8)
