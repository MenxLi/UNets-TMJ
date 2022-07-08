# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: TMJDataset.py                                          | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import os, random
import torch
from torchvision import transforms
import torchvision.transforms.functional as torchTF
import torch.nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from monsoonToolBox.misc.piclkeUtils import readPickle
from monsoonToolBox.misc import divideChunks
import numpy as np
from ..config import TEMP_DIR

train_pickle = os.path.join(TEMP_DIR, "data-train.pkl")
test_pickle = os.path.join(TEMP_DIR, "data-test.pkl")


class TMJDataset2D(Dataset):
    NUM_CLASSES = 4

    def __init__(self, imgs, msks, use_augmentation=True) -> None:
        super().__init__()
        self.imgs = self._flatten(imgs)
        self.msks = self._flatten(msks)
        self.augmentation = use_augmentation

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx][np.newaxis, ...], dtype=torch.float32)
        lbl = torch.tensor(self.msks[idx]).long()
        if self.augmentation:
            return self.randomTransform(img, lbl)
        else:
            return img, lbl

    def _flatten(self, arrs):
        arr = arrs[0]
        for i in range(1, len(arrs)):
            arr = np.concatenate((arr, arrs[i]))
        return arr

    def randomTransform(self, img, lbl):
        _random_transforms = [
            self._randomAdjustBright,
            # self._randomAdjustContrast,
            self._randomRotate,
            self._randomAffine,
            self._flipLeftRight,
        ]
        # n_max_transforms = len(_random_transforms)
        n_max_transforms = 2
        n_transforms = random.randint(0, n_max_transforms)
        transforms = random.sample(_random_transforms, n_transforms)
        for _transform in transforms:
            if random.random() > 0.5:
                img, lbl = _transform(img, lbl)
        return img, lbl

    def _randomRotate(self, img, lbl, degree=30):
        angle = random.randint(-degree, degree)
        image = torchTF.rotate(img, angle)
        segmentation = torchTF.rotate(lbl.unsqueeze(0), angle)
        # more transforms ...
        return image, segmentation[0]

    def _randomAdjustBright(self, img, lbl):
        factor = 2 * random.random() - 1
        factor = (1 / 4) ** factor  # factor \in [1/4, 4]
        image = torchTF.adjust_brightness(img, factor)
        return image, lbl

    def _randomAdjustContrast(self, img, lbl):
        factor = 2 * random.random() - 1
        factor = (1 / 2) ** factor  # factor \in [1/2, 2]
        image = torchTF.adjust_contrast(img, factor)
        return image, lbl

    def _randomAffine(self, img, lbl):
        shear = random.random() * 15
        scale = random.random() * 0.25 + 1
        trans_x = np.random.randint(-50, 50)
        trans_y = np.random.randint(-50, 50)
        angle = random.randint(-10, 10)
        img = torchTF.affine(
            img, angle=angle, translate=(trans_x, trans_y), shear=shear, scale=scale
        )
        lbl = torchTF.affine(
            lbl.unsqueeze(0),
            angle=angle,
            translate=(trans_x, trans_y),
            shear=shear,
            scale=scale,
        )
        return img, lbl[0]

    def _flipLeftRight(self, img, lbl):
        img = transforms.RandomHorizontalFlip(p=1.0)(img)
        lbl = transforms.RandomHorizontalFlip(p=1.0)(lbl)
        return img, lbl

    def _randomCrop(self, img, lbl):
        pass
