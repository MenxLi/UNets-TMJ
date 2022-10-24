import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    smooth = 1e-6

    def __init__(self, gamma=2, weights=None, from_logits=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.from_logits = from_logits
        self.class_average = True
        if not weights is None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
        self.weights = weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        input tesor of shape = (N, C, ...)
        target tensor of shape = (N, ...), 0 or 1
        """
        _device = targets.device
        N = inputs.size(0)
        C = inputs.size(1)
        n_class = C
        if self.weights is None:
            weights = torch.ones(n_class)
        else:
            weights = self.weights
        weights = weights.to(_device)
        weights = weights / weights.sum()
        assert (
            weights.size(0) == n_class
        ), "weights should have the same length as classes dim"
        weights = weights.repeat(N)
        weights_shape = torch.ones(len(inputs.size()), dtype=int)
        weights_shape[0] = N
        weights_shape[1] = n_class
        weights = weights.data.view(*weights_shape)

        if self.from_logits:
            y_pred = F.softmax(inputs, dim=1)
        else:
            y_pred = inputs
        y_true = oneHot(targets, n_classes=n_class, device=_device)
        assert (
            y_true.shape == y_pred.shape
        ), "y_true and y_pred should have the same shape"
        # y == 1
        loss_true = (
            -torch.pow(1 - y_pred, self.gamma)
            * y_true
            * torch.log(y_pred + self.smooth)
        )
        # y == 0
        loss_false = (
            -torch.pow(y_pred, self.gamma)
            * (1 - y_true)
            * torch.log(1 - y_pred + self.smooth)
        )
        loss = loss_true + loss_false
        loss = loss * weights
        loss = loss.sum(dim=1)
        return loss.mean()


def oneHot(targets, n_classes, device="cpu"):
    """Targets: shape - [N, H, W]"""
    targets_extend = targets.clone()
    targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
    one_hot = (
        torch.FloatTensor(
            targets_extend.size(0),
            n_classes,
            targets_extend.size(2),
            targets_extend.size(3),
        )
        .zero_()
        .to(device)
    )
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot
