import os, json
from typing import Literal, List, Dict
import numpy as np
from .config import TEMP_DIR


def imAndMsk(npz_file, mask_from: Literal["masks", "labels"]):
    data = np.load(npz_file, allow_pickle=True)
    imgs = data["imgs"]
    return imgs, data[mask_from]


def filterValid(fname, valid_label_source_names) -> Dict[str, int]:
    global DATA_PATH
    f_path = os.path.join(DATA_PATH, fname)
    idx_label_path = os.path.join(
        DATA_PATH, "LABEL_IDX_" + fname.replace(".npz", ".json")
    )
    with open(idx_label_path, "r") as fp:
        idx_label = json.load(fp)
    valid_idx = dict()
    for idx, label in idx_label.items():
        if label in valid_label_source_names:
            valid_idx[label] = int(idx)
    return valid_idx


def getCommon(fnames: List[str]) -> List[str]:
    sets = []
    for fname in fnames:
        idx_label_path = os.path.join(
            DATA_PATH, "LABEL_IDX_" + fname.replace(".npz", ".json")
        )
        with open(idx_label_path, "r") as fp:
            idx_label = json.load(fp)
        sets.append(set(idx_label.values()))
    return sets[0].intersection(*sets[1:])


def getValidMasks(fname, mask_key, common_files) -> Dict[str, np.ndarray]:
    global DATA_PATH
    f_path = os.path.join(DATA_PATH, fname)
    ims, masks = imAndMsk(f_path, mask_key)
    valid_keys = filterValid(fname, common_files)
    out = dict()
    for k, v in valid_keys.items():
        out[k] = masks[v]
    return out


if __name__ == "__main__":
    DATA_PATH = "/home/monsoon/Documents/Data/TMJ-ML/Results"
    R1_R0_fname = "result_R0-R1.npz"
    R1_R1_fname = "result_manual-intra.npz"
    R1_R2_fname = "result_R1-R2.npz"
    R1_R3_fname = "result_R1-R3_new.npz"
    R1_upp_fname = "result-UPP-best.npz"
    R1_nnu_fname = "results_unet_1000epochs_best.npz"
    #  R1_R0_path = os.path.join(DATA_PATH, R1_R0_fname)
    #  R1_R1_path = os.path.join(DATA_PATH, R1_R1_fname)
    #  R1_R2_path = os.path.join(DATA_PATH, R1_R2_fname)
    #  R1_R3_path = os.path.join(DATA_PATH, R1_R3_fname)
    #  R1_upp_path = os.path.join(DATA_PATH,R1_upp_fname )
    #  R1_nnu_path = os.path.join(DATA_PATH,R1_nnu_fname)

    common_files = getCommon([R1_R0_fname, R1_R2_fname, R1_R3_fname, R1_upp_fname])
    print(f"{len(common_files)} in total")
    print(common_files)

    #  R0_ims, R0_masks = imAndMsk(R1_R0_path, "masks")
    #  R1_ims, R1_masks = imAndMsk(R1_R0_path, "labels")
    #  R2_ims, R2_masks = imAndMsk(R1_R2_path, "labels")
    #  R3_ims, R3_masks = imAndMsk(R1_R3_path, "labels")
    #  upp_ims, upp_masks = imAndMsk(R1_upp_path, "masks")
    #  nnu_ims, nnu_masks = imAndMsk(R1_nnu_path, "masks")

    # get valid images
    f_path = os.path.join(DATA_PATH, R1_R0_fname)
    valid_keys = filterValid(R1_R0_fname, common_files)
    ims, _ = imAndMsk(f_path, "masks")
    im_valid = dict()
    for k, v in valid_keys.items():
        im_valid[k] = ims[v]

    R0_valid_masks = getValidMasks(R1_R0_fname, "masks", common_files)
    R1_valid_masks = getValidMasks(R1_R0_fname, "labels", common_files)
    R2_valid_masks = getValidMasks(R1_R2_fname, "labels", common_files)
    R3_valid_masks = getValidMasks(R1_R3_fname, "labels", common_files)
    upp_valid_masks = getValidMasks(R1_upp_fname, "masks", common_files)
    nnu_valid_masks = getValidMasks(R1_nnu_fname, "masks", common_files)

    assert len(R0_valid_masks) == len(R1_valid_masks)
    assert len(R1_valid_masks) == len(R2_valid_masks)
    assert len(R2_valid_masks) == len(R3_valid_masks)
    assert len(nnu_valid_masks) == len(upp_valid_masks)
    assert len(nnu_valid_masks) == len(R2_valid_masks)
    assert len(im_valid) == len(R2_valid_masks)

    ims_valid = [im_valid[k] for k in valid_keys]
    R0_valid_masks = [R0_valid_masks[k] for k in valid_keys]
    R1_valid_masks = [R1_valid_masks[k] for k in valid_keys]
    R2_valid_masks = [R2_valid_masks[k] for k in valid_keys]
    R3_valid_masks = [R3_valid_masks[k] for k in valid_keys]
    upp_valid_masks = [upp_valid_masks[k] for k in valid_keys]
    nnu_valid_masks = [nnu_valid_masks[k] for k in valid_keys]

    OUTPUT_DIR = os.path.join(TEMP_DIR, "OTHER_DATA")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    np.savez(
        os.path.join(OUTPUT_DIR, "R2-nnu"),
        imgs=ims_valid,
        masks=R2_valid_masks,
        labels=nnu_valid_masks,
    )
    np.savez(
        os.path.join(OUTPUT_DIR, "R3-nnu"),
        imgs=ims_valid,
        masks=R3_valid_masks,
        labels=nnu_valid_masks,
    )
    np.savez(
        os.path.join(OUTPUT_DIR, "R2-upp"),
        imgs=ims_valid,
        masks=R2_valid_masks,
        labels=upp_valid_masks,
    )
    np.savez(
        os.path.join(OUTPUT_DIR, "R3-upp"),
        imgs=ims_valid,
        masks=R3_valid_masks,
        labels=upp_valid_masks,
    )
