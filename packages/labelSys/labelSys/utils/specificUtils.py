#
# Copyright (c) 2020 Mengxun Li.
#
# This file is part of LabelSys
# (see https://bitbucket.org/Mons00n/mrilabelsys/).
#
"""
Specific utilities for this project
"""

import datetime
import cv2 as cv
import numpy as np
from decimal import Decimal
from ..version import __version__


def createHeader(
    labeler, series, config, time=str(datetime.datetime.now()), spacing=(1, 1, 1)
):
    head_info = {
        "Labeler": labeler,
        "Time": time,
        "Spacing": spacing,
        "Series": series,
        "Config": config,
        "Version": __version__,
    }
    return head_info


def getLabelColors(n_colors):
    colors = []
    for c in range(n_colors):
        colors.append([int(180 / n_colors * c), 255, 255])
    colors = _rearrangeColors(colors)
    colors = cv.cvtColor(np.array([colors], np.uint8), cv.COLOR_HSV2RGB).astype(
        np.float
    )
    colors = colors / 255
    return np.squeeze(colors)


def _rearrangeColors(colors):
    """
    - colors: hsv colors of demension [n, 3], value in opencv range
    """
    new = []
    if len(colors) % 2 != 0:
        new.append([90, 125, 125])
        colors = colors[: len(colors) - 1]
    leading = colors[: int(len(colors) / 2)]
    behind = colors[int(len(colors) / 2) :]
    for i in range(int(len(colors) / 2)):
        new.append(leading[i])
        new.append(behind[i])
    return new


def _printLabelColor(colors):
    for c in colors:
        c = map(lambda x: Decimal(x).quantize(Decimal("0.00")), c)
        c = list(c)
        print("[{}, {}, {}]".format(c[0], c[1], c[2]))
