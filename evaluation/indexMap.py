# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: indexMap.py                                            | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
"""
To check the order of initial labels and assosiate them with generated npz file
"""
from typing import Dict, List, Sequence
import json, sys, os
import numpy as np
from monsoonToolBox.filetools import subDirs
from monsoonToolBox.arraytools import Img2D
from monsoonToolBox.misc import lisJobParallel_, lisJobParallel
from labelSys.utils import LabelSysReader
from immarker.sc import showRGBIms
from skimage.transform import resize
from .config import TEMP_DIR


def normalizeAndResizeImgUINT8(imgs, dst_size):
    imgs = [np.asarray(im) for im in imgs]
    imgs = [resize(img, dst_size) for img in imgs]
    imgs = [Img2D.mapMatUint8(img) for img in imgs]
    imgs = [Img2D.gray2rgb(img) for img in imgs]
    return imgs


def normalizeAndResizeImg(imgs, dst_size):
    imgs = [np.asarray(im) for im in imgs]
    imgs = [resize(img, dst_size) for img in imgs]
    imgs = [Img2D.stretchArr(img, 0, 1) for img in imgs]
    return imgs


def findMatch(images1: List[np.ndarray], images2: List[np.ndarray]) -> List[int]:
    SIZE_FOR_MATCH = (20, 100, 100)
    assert len(images1) <= len(
        images2
    ), "length of image1 should be smaller than length of images2"
    # Normalize
    images1 = normalizeAndResizeImg(images1, SIZE_FOR_MATCH)
    images2 = normalizeAndResizeImg(images2, SIZE_FOR_MATCH)

    def match1To2(ims: Sequence[np.ndarray]) -> List[int]:
        """ims: subset images from images1"""
        out = []
        for im in ims:
            sim = []
            for im2 in images2:
                weight = im - im2
                weight = np.abs(weight)
                sim.append(weight.sum())
            out.append(int(np.argmin(sim)))
        return out

    return lisJobParallel(match1To2, images1, use_buffer=False)


if __name__ == "__main__":
    LABEL_DATA_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Data_total"
    idx_label: Dict[str, str] = {}

    labeled_data = sys.argv[1]
    dirs_to_check = subDirs(LABEL_DATA_PATH)
    output_fname = os.path.basename(labeled_data).replace(".npz", ".json")
    output_fname = "LABEL_IDX_" + output_fname
    OUTPUT_PATH = os.path.join(TEMP_DIR, output_fname)

    reader = LabelSysReader(dirs_to_check)
    images1 = np.load(labeled_data, allow_pickle=True)["imgs"]
    if not os.path.exists(OUTPUT_PATH):
        print("Output to: ", OUTPUT_PATH)
        print("Loading images...")

        #  images2 = [d.images for d in reader]
        def loadReaderTmp(reader):
            return [r.images for r in reader]

        images2 = lisJobParallel_(loadReaderTmp, reader)
        print("Matching...")
        matchIdx2 = findMatch(images1, images2)
        print("Exporting...")
        for i, j in enumerate(matchIdx2):
            idx_label[str(i)] = reader[j].path
        with open(OUTPUT_PATH, "w") as fp:
            json.dump(idx_label, fp, indent=1)
    else:
        print("Using existing ", OUTPUT_PATH)
        with open(OUTPUT_PATH, "r") as fp:
            idx_label = json.load(fp)

    # Check
    idx1 = 0
    imgs1 = images1[idx1]
    imgs1 = normalizeAndResizeImgUINT8(imgs1, (256, 256))
    imgs2 = LabelSysReader([os.path.join(LABEL_DATA_PATH, idx_label[str(idx1)])])[
        0
    ].images
    imgs2 = normalizeAndResizeImgUINT8(imgs2, (256, 256))
    combine = [np.concatenate((img1, img2), axis=1) for img1, img2 in zip(imgs1, imgs2)]
    print(combine[0].shape)
    showRGBIms(combine)
