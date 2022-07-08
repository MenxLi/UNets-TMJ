# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: rawDataPreparation.py                                  | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import json
from typing import Tuple, Union
from labelSys.utils.labelReaderV2 import LabelSysReader
from monsoonToolBox.arraytools.img2d import Img2D
from monsoonToolBox.filetools import *
import skimage.transform
import cv2 as cv
import numpy as np
from progress.bar import Bar
import pickle
from ..config import LBL_NUM, TEMP_DIR, TRAIN_PATH, TEST_PATH

IM_HEIGHT = 256
IM_WIDTH = 256

SEQ_ARR_T = Union[List[np.ndarray], np.ndarray]  # type of sequence of 2D array


def _resize(images: SEQ_ARR_T, masks: SEQ_ARR_T) -> Tuple[SEQ_ARR_T, SEQ_ARR_T]:
    """Resize images and masks

    Args:
            images (np.ndarray): array of 2D images
            masks (np.ndarray): array of 2D masks

    Returns:
            (images_resized, masks_resized)
    """
    images = list(
        map(
            lambda x: skimage.transform.resize(x, (IM_HEIGHT, IM_WIDTH), order=1),
            images,
        )
    )
    masks = list(
        map(
            lambda x: cv.resize(
                x.round().astype(np.uint8),
                (IM_HEIGHT, IM_WIDTH),
                interpolation=cv.INTER_NEAREST,
            ),
            masks,
        )
    )
    return images, masks


def getImgsMsksAndSpacing(ori_data_path: str) -> Tuple[SEQ_ARR_T, SEQ_ARR_T, float]:
    """get images, masks and spacing from one folder

    Args:
            ori_data_path (str): ...

    Returns:
            Tuple[SEQ_ARR_T, SEQ_ARR_T, float]: (imgs, msks, spacing)
    """
    legacy_config = '\
	{"labels": ["Disc", "Condyle", "Eminence"], "label_modes": [0, 1, 1], "label_colors": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], "label_steps": [8, 30, 15], "loading_mode": 0, "default_series": "SAG PD", "2D_magnification": 1, "max_im_height": 384}\
	'
    legacy_config = json.loads(legacy_config)
    reader = LabelSysReader([ori_data_path], legacy_config)
    data = reader[0]
    images = []
    masks = []
    for i in range(len(data)):
        img = data.images[i]
        images.append(img)
        msk = data.grayMask(i, LBL_NUM)
        masks.append(msk)
    images = Img2D.stretchArr(np.array(images), max_val=1)
    images, masks = _resize(images, masks)
    spacing = data.header["Spacing"]
    return images, masks, spacing


def generatePickle(data_path: str, flag: str):
    subdirs = subDirs(data_path)
    bar = Bar("Preparing {} data - ".format(flag), max=len(subdirs))
    imgs = []
    msks = []
    spacings = []
    paths = []
    for dir_p in subdirs:
        _imgs, _msks, _spacing = getImgsMsksAndSpacing(dir_p)
        imgs.append(_imgs)
        msks.append(_msks)
        spacings.append(_spacing)
        paths.append(dir_p)
        bar.next()
    data = {"imgs": imgs, "msks": msks, "spacings": spacings, "paths": paths}
    with open(pJoin(TEMP_DIR, "data-{}.pkl".format(flag)), "wb") as fp:  # Pickling
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished. ({})".format(flag))


def main():
    generatePickle(TRAIN_PATH, "train")
    generatePickle(TEST_PATH, "test")


if __name__ == "__main__":
    main()
