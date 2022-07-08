# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: bool1D.py                                              | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
from .basic import StatBasic, Stat1D
import pprint, textwrap

"""Methods for 1D bool array"""

class Bool1D(Stat1D):
    """Methods for 1d bool array"""
    @staticmethod
    def _getFormattedPercentage(data: np.ndarray, tag: str = "Percentage"):
        percentage = Bool1D._calcPercentage(data)
        string = "{}: \n\t{}".format(tag, percentage)
        return string
    
    @staticmethod
    def _calcPercentage(data: np.ndarray):
        """data: 1D bool|int array of 0|1"""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = data.astype(np.int)
        percentage = data.sum()/len(data)
        return percentage
    
    @staticmethod
    def calcConfusionBinary(y_true: np.ndarray, y_pred:np.ndarray) -> dict:
        """
        Calculate confusion matrix, 
        - y_true and y_pred: 1D array of T/F | 1/0
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        y_true = y_true.astype(np.bool)
        y_pred = y_pred.astype(np.bool)
        d_size = y_true.size
        TP = np.logical_and(y_true, y_pred).astype(np.int).sum()/d_size
        TN = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).astype(np.int).sum()/d_size
        FN = np.logical_and(y_true, np.logical_not(y_pred)).astype(np.int).sum()/d_size
        FP = np.logical_and(np.logical_not(y_true), y_pred).astype(np.int).sum()/d_size
        confusion = {
            "TP": TP,
            "TN": TN,
            "FN": FN,
            "FP":FP
        }
        return confusion
    
    @staticmethod
    def calcConfusionBinaryPrint(y_true: np.ndarray, y_pred:np.ndarray, tag = "Confusion", **pp_kwargs) -> dict:
        """
        Calculate and print confusion matrix, 
        - y_true and y_pred: 1D array of T/F | 1/0
        """
        confusion = Bool1D.calcConfusionBinary(y_true, y_pred)
        print("{}:".format(tag))
        pp = pprint.PrettyPrinter(**pp_kwargs)
        # pp.pprint(Bool1D._convertConfusionMatrixAsDataFrame(confusion))
        indent_prefix = "\t\t"
        # wrapper = textwrap.TextWrapper(initial_indent=indent_prefix, subsequent_indent=indent_prefix)
        _indent = "\t"
        _round = 4
        _init_indent = "\t"
        print("{ii}{i}pred_0{i}pred_1".format(ii = _init_indent, i = _indent))
        print("{ii}true_0{i}{TN}{i}{FP}".format(ii = _init_indent, i = _indent, TN = round(confusion["TN"], _round), FP = round(confusion["FP"], _round)))
        print("{ii}true_1{i}{FN}{i}{TP}".format(ii = _init_indent, i = _indent, FN = round(confusion["FN"], _round), TP = round(confusion["TP"], _round)))
        # text = "\tpred_0\tpred_1\ntrue_0\t{TN}\t{FP}\ntrue_1\t{FN}\t{TP}".format(TN = confusion["TN"], TP = confusion["TP"], FN = confusion["FN"], FP = confusion["FP"])
        # print(wrapper.fill(text))
        return confusion

    @staticmethod
    def calcConfusion(y_true: np.ndarray, y_pred:np.ndarray, n_classes:Union[None, int] = None) -> np.ndarray:
        """Calculate confusion matrix on 2 given 1D class arrays (int)
        The first class of the given arrays should be 0

        Args:
            y_true (np.ndarray): 1D array
            y_pred (np.ndarray): 1D array
            n_classes (Union[None, int], optional): number of classes. Defaults to None for auto-infer.

        Returns:
            np.ndarray: 2D array of the counts for each pair
        """
        assert len(y_true) == len(y_pred), "Length mismatch between two inputs for calcConfusion"
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        if n_classes is None:
            n_classes = np.max((np.max(y_true), np.max(y_pred))) + 1

        result = np.zeros(shape = (n_classes, n_classes), dtype=int)
        for i in range(n_classes):
            for j in range(n_classes):
                _y_true = y_true == i
                _y_pred = y_pred == j
                result[i][j] = np.sum(np.logical_and(_y_true, _y_pred))
        return result

    @staticmethod
    def calcConfusionDF(y_true: np.ndarray, y_pred:np.ndarray, n_classes:Union[None, int] = None, legend: Union[List[str], None] = None) -> pd.DataFrame:
        """Calculate confusion matrix on 2 given 1D class arrays (int)
        The first class of the given arrays should be 0

        Args:
            y_true (np.ndarray): 1D array
            y_pred (np.ndarray): 1D array
            n_classes (Union[None, int], optional): number of classes. Defaults to None for auto-infer.
            legend (Union[None, List[str]], optional): name for each class. Defaults to None for range(n_classes).

        Returns:
            pd.DataFrame
        """
        if n_classes is None:
            n_classes = np.max((np.max(y_true), np.max(y_pred))) + 1
        if legend == None:
            legend = list(range(n_classes))
        confusion = Bool1D.calcConfusion(y_true, y_pred, n_classes)
        data = dict()
        for name_, data_ in zip(legend, confusion.T):
            data[name_] = data_
        df = pd.DataFrame(data, index=legend)
        return df


    @staticmethod
    def plotConfusion(y_true, y_pred, save: str = None, labels: List[str] = None, **sns_heatmap_params):
        # https://datatofish.com/confusion-matrix-python/
        import seaborn as sn
        # if not isinstance(y_true, np.ndarray):
            # y_true = np.array(y_true).astype(np.int)
        # if not isinstance(y_pred, np.ndarray):
            # y_pred = np.array(y_pred).astype(np.int)
        # data = {'y_Actual': y_true,
        # 'y_Predicted': y_pred }
        # df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        # confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], \
            # rownames=['Actual'], colnames=['Predicted'], margins = False)

        confusion_matrix = Bool1D.calcConfusionDF(y_true, y_pred)
        sn.heatmap(confusion_matrix, annot=True, xticklabels = labels, yticklabels=labels, **sns_heatmap_params)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        if not save:
            plt.show()
        else:
            plt.savefig(save)

    @staticmethod
    def _convertConfusionMatrixAsDataFrame(confusion):
        data = {"pred_0": pd.Series([confusion["TN"], confusion["FN"]], index = ["true_0", "true_1"]),
                "pred_1": pd.Series([confusion["FP"], confusion["TP"]], index = ["true_0", "true_1"])}
        df = pd.DataFrame(data)
        return df
