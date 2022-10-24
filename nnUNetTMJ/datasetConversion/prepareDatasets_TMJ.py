from ..utils.npImgTools import stretchArr
import os
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.file_conversions import convert_3d_tiff_to_nifti
import skimage.transform, tifffile
import cv2 as cv
import numpy as np
from progress.bar import Bar

# from .labelReader import readOneFolderWithSingleMask
from labelSys.utils.labelReaderV2 import LabelSysReader
from ..config import (
    NUM_LBL,
    TEMP_DIR,
    TRAIN_PATH,
    TEST_PATH,
    LBL_NUM,
)
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir

"""Reference:
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task120_Massachusetts_RoadSegm.py"""

img_tiff_path = os.path.join(TEMP_DIR, "temp_img.tif")
msk_tiff_path = os.path.join(TEMP_DIR, "temp_msk.tif")
IM_HEIGHT = 256
IM_WIDTH = 256


def _generateData(
    ori_data_path: str, out_img_dir: str, out_seg_dir: str, unique_name: str
):
    global img_tiff_path, msk_tiff_path
    legacy_config = '\
		{"labels": ["Disc", "Condyle", "Eminence"], "label_modes": [0, 1, 1], "label_colors": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], "label_steps": [8, 30, 15], "loading_mode": 0, "default_series": "SAG PD", "2D_magnification": 1, "max_im_height": 512}\
		'
    legacy_config = json.loads(legacy_config)
    reader = LabelSysReader([ori_data_path], legacy_config)
    data = reader[0]
    imgs = data.images
    msks = [data.grayMask(i, LBL_NUM) for i in range(len(data.masks))]
    header = data.header
    imgs, msks = _resize(imgs, msks)
    imgs = stretchArr(np.array(imgs), max_val=1)
    spacing = header["Spacing"]
    tifffile.imsave(img_tiff_path, imgs)
    tifffile.imsave(msk_tiff_path, msks)
    output_img_file = join(out_img_dir, unique_name)
    output_seg_file = join(out_seg_dir, unique_name)

    convert_3d_tiff_to_nifti(
        [img_tiff_path], output_img_file, spacing=spacing, is_seg=False
    )
    convert_3d_tiff_to_nifti(
        [msk_tiff_path], output_seg_file, spacing=spacing, is_seg=True
    )


def _resize(images: np.ndarray, masks: np.ndarray):
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


def prepareDatasets():
    task_name = "Task501_TMJSeg"
    print(nnUNet_raw_data)
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # Prepare train dataset
    train_data_paths = []
    for i in os.listdir(TRAIN_PATH):
        p_ = os.path.join(TRAIN_PATH, i)
        if os.path.isdir(p_):
            train_data_paths.append(p_)
    bar = Bar("Preparing train data ", max=len(train_data_paths))
    for i in range(len(train_data_paths)):
        unique_name = "{:03d}".format(i)
        _generateData(
            train_data_paths[i], target_imagesTr, target_labelsTr, unique_name
        )
        bar.next()

    # Prepare test dataset
    test_data_paths = []
    for i in os.listdir(TEST_PATH):
        p_ = os.path.join(TEST_PATH, i)
        if os.path.isdir(p_):
            test_data_paths.append(p_)
    bar = Bar("Preparing test data ", max=len(test_data_paths))
    for i in range(len(test_data_paths)):
        unique_name = "TMJ_{:03d}".format(i)
        _generateData(test_data_paths[i], target_imagesTs, target_labelsTs, unique_name)
        bar.next()
    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        tuple(["MRI"]),
        labels=NUM_LBL,
        dataset_name=task_name,
    )


if __name__ == "__main__":
    prepareDatasets()
