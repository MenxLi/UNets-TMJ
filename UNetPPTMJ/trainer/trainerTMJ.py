import os
import numpy as np
import torch.optim
import torch.nn as nn
from .trainerVanilla import TrainerVanilla
from ..losses import DiceLoss, FusionLoss, FocalLoss
from ..learningRate import PolyLR, CyclicLR, WarmUpLR, CosineLR
import segmentation_models_pytorch as smp
from ..config import TEMP_DIR
from monsoonToolBox.arraytools.draw2d import Drawer2D
from monsoonToolBox.filetools import *
import matplotlib.pyplot as plt


class TrainerTMJ(TrainerVanilla):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.save_dir = pJoin(TEMP_DIR, "model")
        self.save_every = 5
        self.prob_every = 5

        in_channels = 1
        n_classes = 4

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            in_channels=in_channels,
            classes=n_classes,
            activation=None,
        )
        self.total_epochs = 1000
        self.batch_size = 36
        self.base_lr = 1e-4
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.base_lr, weight_decay=0.01
        )
        loss_fn0 = DiceLoss(weights=[0.1, 5, 1, 1], use_softmax=True)
        loss_fn1 = FocalLoss(gamma=3)
        self.loss_fn = FusionLoss(
            [loss_fn0, loss_fn1], weights=[0.5, 0.5], device=self.device
        )

    def getLr(self, epochs: int, total_epochs: int):
        base_instance = CosineLR()
        cycle_instance = CyclicLR(
            cycle_at=[50, 200], lr_snippet=base_instance, decay=[0.5, 0.1]
        )
        warmup_instance = WarmUpLR(lr_instance=cycle_instance, warmup_frac=0.05)
        self.lr_instance = warmup_instance
        return self.lr_instance(epochs, total_epochs, self.base_lr)

    def serveProbImgs(self, img, msk):
        """
        - img of shape (1, H, W)
        - msk: of shape (H, W)
        """
        self.prob_img = img
        self.prob_msk = msk

    def onTrainEpochStart(self, **kwargs) -> None:
        if self.epoch % self.prob_every == 0:
            self.probImg()
        return super().onTrainEpochStart(**kwargs)

    def probImg(self, flag=""):
        if hasattr(self, "prob_img"):
            img = self.prob_img[np.newaxis, ...]
            out = self.model(img.to(self.device))
            pred_msk = out.argmax(dim=1)[0]
            im = self.prob_img[0]
            im = np.array(im.cpu())
            pred_msk = np.array(pred_msk.cpu())
            prob_msk = np.array(self.prob_msk.cpu())

            color_dict = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 100, 255)}
            compare_im = Drawer2D.visualCompareSegmentations(
                im,
                [prob_msk, pred_msk],
                color_dict=color_dict,
                alpha=0.5,
                tags=["Ground truth", "Model prediction"],
            )

            fname = pJoin(TEMP_DIR, "ProbImgs")
            if not os.path.exists(fname):
                os.mkdir(fname)
            fname = pJoin(fname, "epoch-{}{}.png".format(self.epoch, flag))
            plt.imsave(fname, compare_im)
