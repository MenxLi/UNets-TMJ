from os import pread
import os
import typing, random
from typing import Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin

def calcROC(raw_output: np.ndarray, ground_truth: np.ndarray, step: Union[List[float], float] = 0.05) -> List[Tuple[float, float]]:
    """Calculate points in the ROC curve.

    Args:
        raw_output (np.ndarray): array(float), ranged between 0-1, the raw output of the prediction
        ground_truth (np.ndarray): array(int), valued as 0 or 1, the ground prediction with 1 indicate True
        step (Union[List[float], float], optional): step when calculating the points, can be a list or float, ranged between 0-1. Defaults to 0.05.

    Returns:
        List[Tuple[float, float]]: points of (x,y) in ROC curve
    """
    if not isinstance(raw_output, np.ndarray):
        raw_output = np.array(raw_output)
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)
    if isinstance(step, float):
        n_step = int(1//step)
        step = np.linspace(0, 1, n_step)
    assert raw_output.shape == ground_truth.shape, "raw_output should have the same shape as ground_truth"
    
    smooth = 1e-7
    points: List[Tuple[float, float]] = list()
    gt = ground_truth.astype(bool)
    for i in step:
        pred = raw_output > i

        tp = np.logical_and(pred, gt).sum()
        p = gt.sum()
        fp = np.logical_and(pred, 1-gt).sum()
        n = (1-gt).sum()

        tpr = tp / (p + smooth) # True positive rate, sensitivity, vertical axis
        fpr = fp / (n + smooth) # False positive rate, 1 - sensitivity, horizontal axis

        points.append((fpr, tpr))
    return points

def calcAUC(points: List[Tuple[float, float]]):
    pts = sorted(points, key= lambda x: x[0])
    auc = 0
    for i in range(1, len(pts)):
        auc += (pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2
    return auc

def multiclassROC(raw_output: np.ndarray, ground_truth: np.ndarray, step: Union[List[float], float] = 0.05, average: bool = False,) -> List[List[Tuple[float, float]]]:
    """Calculate and plot the multi-class ROC curve.

    Args:
        raw_output (np.ndarray): array(float), ranged between 0-1, the raw output of the prediction, e.g for 2 class: [[0.01, 0.83], [0.98, 0.08], ...]
        ground_truth (np.ndarray): array(int), valued as 0 or 1, the ground prediction with 1 indicate True, e.g for 2 class: [[0, 1], [1, 0], ...]
        step (Union[List[float], float], optional): step when calculating the points, can be a list or float, ranged between 0-1. Defaults to 0.05
        average (bool, optional): output average roc curve points.

    Returns:
        List[List[Tuple[float, float]]]: Lists of all points in ROC curves
    """
    if not isinstance(raw_output, np.ndarray):
        raw_output = np.array(raw_output)
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)
    n_class = ground_truth.shape[-1]
    roc_points: List[List[Tuple[float, float]]] = []

    for c in range(n_class):
        _pred = raw_output[..., c]
        _gt = ground_truth[..., c]

        roc_pts = calcROC(_pred, _gt, step)
        roc_points.append(roc_pts)
    if average:
        roc_points = [np.array(roc_points).mean(axis=0)]
    return roc_points

def plotROCs(roc_points: List[List[Tuple[float, float]]], 
            colors: Union[List[str], None] = None, 
            legends: Union[List[str], None] = None,
            fmt: str = "-",
            plt_params: dict = {},
            save_fname: Union[None, str] = None):
    n_class = len(roc_points)
    if colors is None:
        colors = [randomHexColor() for i in range(n_class)]
    if legends is None:
        legends = [f"Class - {str(i)}" for i in range(n_class)]
    for roc_pts, _legend, _color in zip(roc_points, legends, colors):
        roc_pts = np.array(roc_pts)
        x = roc_pts[:, 0]
        y = roc_pts[:, 1]
        plt.plot(x, y, fmt, label = _legend, color = _color, **plt_params)
    plt.plot((0, 1), (0, 1), linestyle = "dashed", color = "gray")
    plt.legend()
    if save_fname:
        plt.savefig(save_fname, dpi = 300, pad_inches = 0)
    plt.show()
    return

def randomHexColor():
    random_number = random.randint(0,16777215)
    hex_number = str(hex(random_number))
    hex_number ='#'+ hex_number[2:]
    return hex_number