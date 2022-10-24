"""
Preprocess masks to deal with the undefined ends of condyle and eminence
Condyle is supposed to be closed contour, eminence is supposed to be opencontour
"""

import numpy as np
import cv2 as cv
from .config import LBL_NUM, NUM_LBL


def processCondyle(condyle_mask0, condyle_mask1, return_type="open contour"):
    """
    Cut the condyle at certain horizontal level to unify the ends of the labels
    ----args:
    condyle_mask (np array): mask that contain only the condyle, 2D array
    return_type (str):
         "open contour": masks of open contours with thickness 1px
         ~"close contour": masks of close contours~
         ~"points": all points on the contour~
    """
    squeeze_0 = condyle_mask0.astype(np.bool).any(axis=1)
    squeeze_1 = condyle_mask1.astype(np.bool).any(axis=1)
    t_ = np.logical_and(squeeze_0, squeeze_1)
    start_line, end_line = _findRange(t_)
    end_line_refine = int((end_line - start_line) * 3 / 4 + start_line)
    m0 = condyle_mask0.copy()
    m0[end_line_refine:, :] = 0
    m1 = condyle_mask1.copy()
    m1[end_line_refine:, :] = 0
    if return_type == "close contour":
        return m0, m1
    else:
        raise NotImplementedError("Incorrect return_type in preprocessCondyle")


def processEminence(eminence_mask0, eminence_mask1, return_type="open contour"):
    """
    Cut the eminence at certain vertical levels to unify the ends of the labels
    ----args:
    eminence_mask (np array): mask that contain only the eminence, 2D array
    return_type (str):
         "open contour": masks of open contours with thickness 1px
         "points": all points on the contour
    """
    squeeze_0 = eminence_mask0.astype(np.bool).any(axis=0)
    squeeze_1 = eminence_mask1.astype(np.bool).any(axis=0)
    t_ = np.logical_and(squeeze_0, squeeze_1)
    start_line, end_line = _findRange(t_)
    m0 = eminence_mask0.copy()
    m1 = eminence_mask1.copy()
    m0[:, :start_line] = 0
    m0[:, end_line:] = 0
    m1[:, :start_line] = 0
    m1[:, end_line:] = 0
    pts0 = _averagePoints(_findPoints(m0, direction="vertical"))
    pts1 = _averagePoints(_findPoints(m1, direction="vertical"))
    if return_type == "open contour":
        m0_ = np.zeros(m0.shape, np.uint8)
        m1_ = np.zeros(m1.shape, np.uint8)
        _drawLines(m0_, pts0)
        _drawLines(m1_, pts1)
        return m0_, m1_
    elif return_type == "points":
        return pts0, pts1
    else:
        raise NameError("Incorrect return_type in preprocessEminence")


def _findRange(seq):
    start = None
    end = None
    for i in range(len(seq)):
        if seq[i] == 1:
            start = i
            break
    for i in range(len(seq)):
        if seq[len(seq) - 1 - i] == 1:
            end = len(seq) - 1 - i
            break
    return start, end


def _findPoints(mask, direction="horizontal"):
    """
    return (x, y) - cv coordinate
    """
    points = []
    if direction == "horizontal":
        for i in range(len(mask)):
            line = mask[i, :]
            if (line == 0).all():
                continue
            s, e = _findRange(line)
            if s == e:
                points.append([(s, i)])
            else:
                points.append([(s, i), (e, i)])
    elif direction == "vertical":
        for j in range(len(mask[0])):
            line = mask[:, j]
            if (line == 0).all():
                continue
            s, e = _findRange(line)
            if s == e:
                points.append([(j, s)])
            else:
                points.append([(j, s), (j, e)])
    else:
        raise Exception("In correct direction in _findPoints")
    return points


def _averagePoints(pts):
    """average points set if more than one point was found - for eminence"""
    for i in range(len(pts)):
        if len(pts[i]) > 1:
            t = np.array(pts[i])
            pts[i] = tuple(t.mean(axis=0).flatten().astype(int))
        else:
            pts[i] = pts[i][0]
    return np.array(pts)


def _drawLines(img, pts):
    for i in range(1, len(pts)):
        cv.line(img, tuple(pts[i - 1]), tuple(pts[i]), 1, 1)
    return img


def postProc(mask1, mask2):
    condyle_masks = (mask1 == LBL_NUM["Condyle"], mask2 == LBL_NUM["Condyle"])
    if not ((condyle_masks[0] == 0).all() or (condyle_masks[1] == 0).all()):
        condyle_masks = processCondyle(*condyle_masks, return_type="close contour")
    eminence_masks = (mask1 == LBL_NUM["Eminence"], mask2 == LBL_NUM["Eminence"])
    if not ((eminence_masks[0] == 0).all() or (eminence_masks[1] == 0).all()):
        eminence_masks = processEminence(*eminence_masks, return_type="open contour")
    disc_masks = (mask1 == LBL_NUM["Disc"], mask2 == LBL_NUM["Disc"])
    masks = {"Condyle": condyle_masks, "Eminence": eminence_masks, "Disc": disc_masks}
    m1 = np.zeros(shape=mask1.shape)
    m2 = np.zeros(shape=mask2.shape)
    for i in [1, 2, 3]:
        m1_, m2_ = masks[NUM_LBL[i]]
        m1 = m1_ * i + m1 * (1 - m1_)
        m2 = m2_ * i + m2 * (1 - m2_)
    return m1.astype(np.int), m2.astype(np.int)


def postProc_old(mask1, mask2):
    condyle_masks = (mask1 == LBL_NUM["Condyle"], mask2 == LBL_NUM["Condyle"])
    condyle_masks = processCondyle(*condyle_masks, return_type="close contour")
    eminence_masks = (mask1 == LBL_NUM["Eminence"], mask2 == LBL_NUM["Eminence"])
    eminence_masks = processEminence(*eminence_masks, return_type="open contour")
    disc_masks = (mask1 == LBL_NUM["Disc"], mask2 == LBL_NUM["Disc"])
    masks = {"Condyle": condyle_masks, "Eminence": eminence_masks, "Disc": disc_masks}
    to_concat = [masks[NUM_LBL[i]] for i in [1, 2, 3]]
    m1 = [m[0][..., np.newaxis] for m in to_concat]
    m2 = [m[1][..., np.newaxis] for m in to_concat]
    return np.concatenate(m1, axis=-1).astype(np.float), np.concatenate(
        m2, axis=-1
    ).astype(np.float)
