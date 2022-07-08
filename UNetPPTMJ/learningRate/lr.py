# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: lr.py                                                  | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
from os import wait
from typing import Any, Callable, List, Union
import numpy as np


class LearningRateAbstract(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, epoch: int, max_epoch: int, base_lr: float) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        return __class__.__name__

    __repr__ = __str__


class WarmUpLR(LearningRateAbstract):
    def __init__(
        self,
        lr_instance: Union[
            LearningRateAbstract, Callable
        ] = lambda epoch, max_epoch, base_lr: float(max_epoch - epoch)
        / max_epoch
        * base_lr,
        warmup_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.lr_instance = lr_instance
        self.warmup_frac = warmup_frac

    def __call__(self, epoch: int, max_epoch: int, base_lr: float) -> None:
        warmup_stop_epoch = int(max_epoch * self.warmup_frac)
        if epoch < warmup_stop_epoch:
            return base_lr * float(epoch) / warmup_stop_epoch
        else:
            _epoch = epoch - warmup_stop_epoch
            _max_epoch = max_epoch - warmup_stop_epoch
            return self.lr_instance(_epoch, _max_epoch, base_lr)

    def __str__(self):
        return (
            __class__.__name__
            + "(Warm_up start at: {})".format(self.warmup_frac)
            + " <- "
            + self.lr_instance.__str__()
        )


class PolyLR(LearningRateAbstract):
    def __init__(self, power=2) -> None:
        super().__init__()
        self.power = power

    def __call__(self, epoch: int, max_epoch: int, base_lr: float) -> None:
        return base_lr * (1 - (epoch / max_epoch)) ** self.power

    def __str__(self) -> str:
        return __class__.__name__ + "(power = {})".format(self.power)


class CosineLR(LearningRateAbstract):
    def __call__(self, epoch: int, max_epoch: int, base_lr: float) -> None:
        return 1 / 2 * (np.cos(np.pi * epoch / max_epoch) + 1) * base_lr

    def __str__(self) -> str:
        return __class__.__name__


class CyclicLR(LearningRateAbstract):
    def __init__(
        self,
        cycle_at: List[int] = [0],
        lr_snippet: Union[
            LearningRateAbstract, Callable
        ] = lambda epoch, max_epoch, base_lr: float(max_epoch - epoch)
        / max_epoch
        * base_lr,
        decay: Union[None, float, List[float]] = None,
    ) -> None:
        """
        - cycle_at: list of epochs that you want to start cycle
        - decay: base_lr decay at each cycle, should be None or the same length as cycle_at
        """
        super().__init__()
        self.snippet = lr_snippet
        if decay is None:
            decay = [1.0] * len(cycle_at)
        elif isinstance(decay, float):
            decay = [decay ** (i + 1) for i in range(len(cycle_at))]
        else:
            assert len(decay) == len(
                cycle_at
            ), "cycle_at and decay should have the same length for CyclicLR"

        if cycle_at[0] != 0:
            self.cycle_decay = [(0, 1)]
            self.cycle_decay += [(c, d) for c, d in zip(cycle_at, decay)]
        else:
            self.cycle_decay = [(c, d) for c, d in zip(cycle_at, decay)]
        self.cycle_decay.sort(key=lambda x: x[0])

    def __call__(self, epoch: int, max_epoch: int, base_lr: float) -> None:
        for i in range(len(self.cycle_decay) - 1):
            if epoch >= self.cycle_decay[i][0] and epoch < self.cycle_decay[i + 1][0]:
                _epoch = epoch - self.cycle_decay[i][0]
                _base_lr = base_lr * self.cycle_decay[i][1]
                _max_epoch = self.cycle_decay[i + 1][0] - self.cycle_decay[i][0]
                break
        if epoch >= self.cycle_decay[-1][0]:
            _epoch = epoch - self.cycle_decay[-1][0]
            _base_lr = base_lr * self.cycle_decay[-1][1]
            _max_epoch = max_epoch - self.cycle_decay[-1][0]
        return self.snippet(_epoch, _max_epoch, _base_lr)

    def __str__(self) -> str:
        return (
            __class__.__name__
            + "([cycle, decay]: {})".format(self.cycle_decay)
            + " <- "
            + self.snippet.__str__()
        )


# Test
if __name__ == "__main__":
    from .lrUtils import plotLR

    lr_instance = CosineLR()
    # cycle_instance = CyclicLR(cycle_at=[50, 100, 500], lr_snippet=lr_instance, decay=[0.8, 0.5, 0.1])
    # warmup_instance = WarmUpLR(lr_instance=cycle_instance)
    warmup_instance = WarmUpLR(lr_instance=lr_instance, warmup_frac=0.05)
    cycle_instance = CyclicLR(
        cycle_at=[100, 500], lr_snippet=warmup_instance, decay=[0.5, 0.1]
    )
    instance = cycle_instance

    cosine_instance = CosineLR()
    cycle_instance = CyclicLR(
        cycle_at=[50, 200], lr_snippet=cosine_instance, decay=[0.5, 0.1]
    )
    warmup_instance = WarmUpLR(lr_instance=cycle_instance, warmup_frac=0.05)
    instance = warmup_instance
    print(instance)
    plotLR(instance, max_epoch=500)
