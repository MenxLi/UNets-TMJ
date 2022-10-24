from matplotlib import use
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, softmax
import numpy as np


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        """
        input and targets of shape:
                (N, C, H, W)
        """
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 0.001
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weights=None, use_softmax=False, **kwargs):
        super().__init__()
        self.weights = weights
        self.kwargs = kwargs
        self.softmax = use_softmax

    def forward(self, input, target):
        """
        input tesor of shape = (N, C, H, W)
        target tensor of shape = (N, H, W)
        """
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        nclass = input.shape[1]
        if self.weights is None:
            weights = torch.tensor(np.ones(nclass))
        else:
            weights = torch.tensor(self.weights)
        weights = weights / weights.sum()
        target = oneHot(target, n_classes=nclass)

        assert input.shape == target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        if self.softmax:
            logits = softmax(input, dim=1)
        else:
            logits = input

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(nclass):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i]) * weights[i]
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss


def oneHot(targets, n_classes, gpu=True):
    """Targets: shape - [N, H, W]"""
    targets_extend = targets.clone()
    targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
    if gpu:
        one_hot = torch.cuda.FloatTensor(
            targets_extend.size(0),
            n_classes,
            targets_extend.size(2),
            targets_extend.size(3),
        ).zero_()
    else:
        one_hot = torch.FloatTensor(
            targets_extend.size(0),
            n_classes,
            targets_extend.size(2),
            targets_extend.size(3),
        ).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot
