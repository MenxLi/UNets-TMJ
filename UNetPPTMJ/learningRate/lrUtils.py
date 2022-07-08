# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: lrUtils.py                                             | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
from matplotlib import pyplot as plt
import numpy as np


def plotLR(lr_instance, max_epoch=1000):
    lr = [lr_instance(i, max_epoch, 1) for i in range(max_epoch)]
    plt.plot(lr)
    plt.show()
