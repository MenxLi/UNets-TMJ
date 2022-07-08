# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: basic.py                                               | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
from typing import Tuple, Union
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import matplotlib.pyplot as plt


class StatBasic:
    def __init__(self):
        pass
    @staticmethod
    def meanStd(data: np.ndarray):
        mean = np.mean(data)
        std = np.std(data)
        count = len(data)
        return {
                "mean": mean,
                "std": std,
                "count": count
                }
    @staticmethod
    def getFormattedMeanStd(data: np.ndarray, tag = "Mean and Std"):
        statistic = StatBasic.meanStd(data)
        mean_std = "Mean: {mean} | Std: {std} - count: {count}".format(mean = statistic["mean"], std = statistic["std"], count = statistic["count"])
        string = "{}: \n\t{}".format(tag, mean_std)
        return string

class Stat1D(StatBasic):
    @staticmethod
    def washNone(data: Union[np.ndarray, list]) -> Tuple[np.ndarray, int]:
        """
        Delete (wash out) all None items in the list and return residual items and None count
        data - 1D array of number or None
        return data_washed, none_count
        """
        if isinstance(data, list):
            data = np.array(data)
        none_mask = data == None
        none_count = none_mask.sum()
        data_ = np.ma.masked_array(data, none_mask).compressed()
        return data_, none_count
    
    @staticmethod
    def notZeros(data: Union[np.ndarray, list]) -> np.ndarray:
        """
        Check if element in the array is all zeros
        Return 1D bool array of the status of each element
        """
        result = np.zeros(shape = len(data), dtype = bool)
        for i in range(len(data)):
            if (data[i] == 0).all():
                result[i] = True
        return np.logical_not(result)
