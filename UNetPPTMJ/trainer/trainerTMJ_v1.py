# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: trainerTMJ_v1.py                                       | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import torch.optim
import segmentation_models_pytorch as smp

from .trainerTMJ import TrainerTMJ
from ..learningRate import CosineLR, CyclicLR, WarmUpLR, PolyLR
from ..losses import DiceLoss, FocalLoss, FusionLoss


class TrainerTMJ_v1(TrainerTMJ):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        in_channels = 1
        n_classes = 4
        self.model = smp.UnetPlusPlus(
            encoder_name="se_resnext50_32x4d",
            in_channels=in_channels,
            classes=n_classes,
            activation=None,
            decoder_attention_type="scse",
        )
        self.total_epochs = 1000
        self.batch_size = 4
        self.base_lr = 3e-3
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0,
        )

    def getLr(self, epochs: int, total_epochs: int):
        base_instance = PolyLR(power=2)
        cycle_instance = CyclicLR(
            cycle_at=[50, 150, 325, 600], lr_snippet=base_instance, decay=0.2
        )
        warmup_instance = WarmUpLR(lr_instance=cycle_instance, warmup_frac=0.02)
        self.lr_instance = warmup_instance
        return self.lr_instance(epochs, total_epochs, self.base_lr)
