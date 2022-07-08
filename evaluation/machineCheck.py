# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: machineCheck.py                                        | #
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
from typing import Dict, List
import json
import os, sys
import numpy as np
import cv2 as cv
from .machineAnalysis import JSON_SAVE_PATH
from monsoonToolBox.filetools import subDirs
from monsoonToolBox.arraytools import Img2D
from monsoonToolBox.misc import lisJobParallel_
from labelSys.utils import LabelSysReader
from immarker.sc import showRGBIms
from skimage.transform import resize

JSON_SAVE_PATH_REVISED = "/home/monsoon/Documents/Data/TMJ-ML/machine_revised.json"
JSON_LABEL_MACHINE_PTH = JSON_SAVE_PATH_REVISED
TOTAL_DATA_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Data_total"
TRAIN_DATA_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Data_formatted_2/train"
TEST_DATA_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Data_formatted_2/test"
TEST_NPZ_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Results/result-UPP-best.npz"
JSON_LABEL_IDX_PTH = "/home/monsoon/Documents/Data/TMJ-ML/label_npz_idx.json"

with open(JSON_SAVE_PATH_REVISED, "r") as fp:
    dir_machine = json.load(fp)

# Machine distribution
def dataCount(dir_path):
    machine_count: Dict[str, int] = {}
    for d in subDirs(dir_path):
        dir_name = os.path.basename(d)
        machine = dir_machine[dir_name]
        try:
            machine_count[machine] += 1
        except KeyError:
            machine_count[machine] = 1
    return machine_count


def normalizeImg(img):
    img = np.asarray(img)
    img = Img2D.mapMatUint8(img)
    img = [Img2D.gray2rgb(im) for im in img]
    return img


def find_match(dir_path, images: List[np.ndarray]) -> int:
    label = LabelSysReader([dir_path])[0]
    lbl_imgs = np.asarray(label.images)
    lbl_imgs = Img2D.stretchArr(lbl_imgs, 0, 1)
    lbl_imgs_resized = resize(lbl_imgs, SIZE_FOR_MATCH)
    sim = []
    for im in images:
        weight = lbl_imgs_resized - im
        weight = np.abs(weight)
        sim.append(weight.sum())
    return int(np.argmin(sim))


def find_match_p(dir_paths: List[str]) -> List[int]:
    global npz_images_resized
    return [find_match(s, npz_images_resized) for s in dir_paths]


# check
def showIdx(file_idx: int, label_idx: int):
    global dirs_to_check, npz_images
    print("checking idx: ", file_idx, label_idx)
    label = LabelSysReader(dirs_to_check)

    file_images = label[file_idx].images
    file_images = np.asarray(file_images)
    print(file_images.shape)
    file_images = Img2D.mapMatUint8(file_images)
    file_images = [cv.resize(Img2D.gray2rgb(im), (256, 256)) for im in file_images]

    label_images = data["imgs"][label_idx]
    label_images = np.asarray(label_images)
    print(label_images.shape)
    label_images = Img2D.mapMatUint8(label_images)
    label_images = [Img2D.gray2rgb(im) for im in label_images]

    combine = [
        np.concatenate((label_images[i], file_images[i]), axis=0)
        for i in range(len(label_images))
    ]
    print(combine[0].shape)

    showRGBIms(combine)


if __name__ == '__main__':
    # machine distribution
    print("Total")
    print(dataCount(TOTAL_DATA_PATH))
    print("Train")
    print(dataCount(TRAIN_DATA_PATH))
    print("Test")
    print(dataCount(TEST_DATA_PATH))

    # Data matching
    #  label = LabelSysReader(subDirs(TEST_DATA_PATH))

    #  TEST_NPZ_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Results/results_unet_1000epochs_best.npz"
    data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    SIZE_FOR_MATCH = (20, 100, 100)
    npz_images = data["imgs"]
    npz_images_resized = [resize(np.asarray(im), SIZE_FOR_MATCH) for im in npz_images]
    npz_images_resized = [Img2D.stretchArr(im, 0, 1) for im in npz_images_resized]

    label_idx: Dict[str, int] = {}
    dirs_to_check = subDirs(TEST_DATA_PATH)
    data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    if not os.path.exists(JSON_LABEL_IDX_PTH):
        matches = lisJobParallel_(find_match_p, dirs_to_check)
        for d, m in zip(dirs_to_check, matches):
            label_idx[os.path.basename(d)] = m
        with open(JSON_LABEL_IDX_PTH, "w") as fp:
            json.dump(label_idx, fp, indent=1)
    else:
        print("Using existing ", JSON_LABEL_IDX_PTH)
        with open(JSON_LABEL_IDX_PTH, "r") as fp:
            label_idx = json.load(fp)

    ## TEMP
    #  matches = lisJobParallel_(find_match_p, dirs_to_check)
    #  for d, m in zip(dirs_to_check, matches):
    #      label_idx[os.path.basename(d)] = m
    #  #  with open(JSON_LABEL_IDX_PTH, "w") as fp:
    #  #      json.dump(label_idx, fp, indent=1)
    #  with open(JSON_LABEL_IDX_PTH, "r") as fp:
    #      label_idx_old = json.load(fp)
    #  for k, v in label_idx.items():
    #      print(label_idx_old[k] == v)
    #  exit()
    ## TEMP END

    # Check
    check_idx = 8
    file_path = dirs_to_check[check_idx]
    file_base_name = os.path.basename(file_path)
    _label_idx = label_idx[file_base_name]
    showIdx(check_idx, _label_idx)
