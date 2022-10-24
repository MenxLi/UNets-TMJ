from .postprocessMasks import postProc
import sys, os, shutil
import numpy as np
import monsoonToolBox as tbx
from monsoonToolBox.logtools import Timer
from monsoonToolBox.arraytools import Drawer2D
from monsoonToolBox.misc.progress import printProgressBar
from .evalResult import generateCompareImages, batchPostProc_p
from .config import LBL_NUM, TEMP_DIR
import matplotlib.pyplot as plt
import scipy.ndimage
from .postprocessMasks import processEminence


def generateCompareImages(imgs, masks_upp, masks_nnu, labels):
    # Generate compare images
    compare_im_dir = os.path.join(TEMP_DIR, "comapreImg")
    if not os.path.exists(compare_im_dir):
        os.mkdir(compare_im_dir)
    color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 100, 255)}
    print("Generating compare images...")
    dir_count = 0
    for ims, msks_upp, msks_nnu, lbls in zip(imgs, masks_upp, masks_nnu, labels):
        p_dir = os.path.join(compare_im_dir, str(dir_count))
        if os.path.exists(p_dir):
            shutil.rmtree(p_dir)
        os.mkdir(p_dir)
        im_count = 0
        for im, msk_upp, msk_nnu, lbl in zip(ims, msks_upp, msks_nnu, lbls):
            # Original image size of 256x256 was assumed
            im_zoom = scipy.ndimage.zoom(im[64:192, 64:192], 2.0)
            msk_upp_zoom = scipy.ndimage.zoom(msk_upp[64:192, 64:192], 2.0, order=0)
            msk_nnu_zoom = scipy.ndimage.zoom(msk_nnu[64:192, 64:192], 2.0, order=0)
            lbl_zoom = scipy.ndimage.zoom(lbl[64:192, 64:192], 2.0, order=0)
            comp_im = Drawer2D.visualCompareSegmentations(
                im_zoom,
                [lbl_zoom, msk_upp_zoom, msk_nnu_zoom],
                color_dict=color_dict,
                alpha=0.5,
                tags=["Ground truth", "U-Net++", "nnUNet"],
            )
            new_first_im = Drawer2D.addTagToImg(
                Drawer2D.gray2rgb(Drawer2D.mapMatUint8(im)), "Original image"
            )
            comp_im[:, :256, :] = new_first_im
            # comp_im = Drawer2D.visualCompareSegmentations(im, [lbl, msk_upp, msk_nnu], color_dict=color_dict, alpha=0.5, tags=["Ground truth", "U-Net++", "nnUNet"])
            im_path = os.path.join(
                compare_im_dir, str(dir_count), str(im_count) + ".png"
            )
            plt.imsave(im_path, comp_im)
            im_count += 1
        printProgressBar(dir_count, len(imgs))
        dir_count += 1
    print(
        "Finished generating compare images, the images were saved to ", compare_im_dir
    )


def thinEminence(masks):
    """Masks: list of 3D arrays"""
    new_masks = []
    for msks in masks:
        new_msks = []
        for msk in msks:
            condyle_mask = msk == LBL_NUM["Eminence"]
            msk = msk * (1 - condyle_mask)
            condyle_mask, _ = processEminence(condyle_mask, condyle_mask)
            msk = msk + condyle_mask * LBL_NUM["Eminence"]
            new_msks.append(msk)
        new_masks.append(new_msks)
    return new_masks


if __name__ == "__main__":
    npz_paths = sys.argv[1:]
    masks_raw = []
    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        imgs_ = imgs_ = data["imgs"]
        masks_ = data["masks"]
        labels_ = data["labels"]

        masks_raw.append(masks_)
    masks_upp = masks_raw[0]
    masks_upp = thinEminence(masks_upp)
    masks_nnu = masks_raw[1]
    masks_nnu = thinEminence(masks_nnu)
    generateCompareImages(imgs_, masks_upp, masks_nnu, labels_)
