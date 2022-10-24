import nibabel as nib
import numpy as np
import sys, os, shutil
import matplotlib.pyplot as plt
from ..config import TEMP_DIR

sys.path.append("/home/monsoon/Documents/Code/toolBox")
from monsoonToolBox.env import getEnvVar
import monsoonToolBox as tbx
from monsoonToolBox.arraytools.draw2d import Drawer2D


def readNibabelToNp(fpath: str):
    img = nib.load(fpath)
    return np.array(img.dataobj)


def getImgsFromNib(p_im):
    img_raw = nib.load(p_im).dataobj[:, :, :]
    img = np.array([img_raw[:, :, i] for i in range(img_raw.shape[2])])
    img_rot = np.rot90(img, k=-1, axes=(1, 2))
    img_revert = np.flip(img_rot, axis=2)
    return img_revert


if __name__ == "__main__":
    nnunet_raw_db = getEnvVar("nnUNet_raw_data_base")
    temp_dir = TEMP_DIR
    nib_labels_path = os.path.join(
        nnunet_raw_db, "nnUNet_raw_data/Task501_TMJSeg/labelsTs"
    )
    nib_imgs_path = os.path.join(
        nnunet_raw_db, "nnUNet_raw_data/Task501_TMJSeg/imagesTs"
    )
    nib_masks_path = os.path.join(temp_dir, "Ensemble_unet_3dfullres_1000epochs")

    npz_path = os.path.join(TEMP_DIR, "results-nnunet.npz")

    nib_masks = tbx.filetools.subFiles(nib_masks_path)
    nib_masks = [x for x in nib_masks if x.endswith(".nii.gz")]
    nib_masks.sort()
    nib_labels = tbx.filetools.subFiles(nib_labels_path)
    nib_labels.sort()
    nib_imgs = tbx.filetools.subFiles(nib_imgs_path)
    nib_imgs.sort()

    imgs = [getImgsFromNib(im_p) for im_p in nib_imgs]
    masks = [getImgsFromNib(msk_p) for msk_p in nib_masks]
    labels = [getImgsFromNib(lbl_p) for lbl_p in nib_labels]

    np.savez(npz_path, imgs=imgs, masks=masks, labels=labels, dtype=np.object)
    print("Success, saved npz file: ", npz_path)
