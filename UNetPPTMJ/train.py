# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: train.py                                               | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #

from monsoonToolBox.misc import divideFold
from monsoonToolBox.filetools import pJoin
import os, argparse
from monsoonToolBox.misc.piclkeUtils import readPickle
from torch.utils.data import DataLoader
from UNetPPTMJ.config import TEMP_DIR
from UNetPPTMJ.dataloading.TMJDataset import TMJDataset2D
from UNetPPTMJ.trainer import trainerTMJ
import matplotlib.pyplot as plt


def main():
    train_pickle = os.path.join(TEMP_DIR, "data-train.pkl")
    test_pickle = os.path.join(TEMP_DIR, "data-test.pkl")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", action="store_true", default=False, help="Continue trainning"
    )
    parser.add_argument("-f", "--fold", action="store", default="all", help="fold")
    args = parser.parse_args()

    print("loading data...")
    fold = args.fold
    model_save_dir = pJoin(TEMP_DIR, "model_f-{}".format(fold))
    data = readPickle(train_pickle)
    imgs = data["imgs"]
    msks = data["msks"]
    if fold == "all":
        data_validate = readPickle(test_pickle)
        train_dataset = TMJDataset2D(imgs=imgs, msks=msks)
        test_dataset = TMJDataset2D(
            imgs=data_validate["imgs"],
            msks=data_validate["msks"],
            use_augmentation=False,
        )
    else:
        fold = int(fold)
        try:
            test_imgs, train_imgs = divideFold(imgs, fold=fold, total_fold=5)
            test_msks, train_msks = divideFold(msks, fold=fold, total_fold=5)
        except ValueError as e:
            # too small data size
            if str(e) == "range() arg 3 must not be zero":
                raise ValueError(
                    "Data size is too small to divide folds, run `train --fold all` instead"
                )
        train_dataset = TMJDataset2D(imgs=train_imgs, msks=train_msks)
        test_dataset = TMJDataset2D(
            imgs=test_imgs, msks=test_msks, use_augmentation=False
        )
    print("Trainning length: ", len(train_dataset))
    print("Test length: ", len(test_dataset))

    # prob image, for monitering the training process
    p_idx = 4
    p_im, p_msk = test_dataset[p_idx]

    n_workers = 0

    print("Starting trainning")
    trainer = trainerTMJ.TrainerTMJ()
    # trainer.batch_size = batch_size # for recording propose
    trainer.feed(train_dataset, test_dataset, shuffle=True, num_workers=n_workers)
    trainer.serveProbImgs(p_im, p_msk)
    trainer.save_dir = model_save_dir
    if args.c:
        if os.path.exists(model_save_dir):
            trainer.loadModel(trainer.save_dir)
        else:
            print("The result folder does not exists, training a new model instead.")
    else:
        if os.path.exists(model_save_dir):
            _continue = input(
                "The model saving dir exists, re-train this fold? (y/yes/<other>): "
            )
            if not (_continue == "y" or _continue == "yes"):
                print("Abort.")
                exit()
    # trainer.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
