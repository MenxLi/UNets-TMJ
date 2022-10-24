import typing
from typing import Union, List, Tuple
import numpy as np
from .arrayBase import Array2D
from .maskNd import MaskNd, MaskEvalNd
from scipy import ndimage
import skfmm
import cv2 as cv
from ..misc import lisJobParallel


class Mask2D(MaskNd, Array2D):
    @staticmethod
    def getCentroid(msk: np.ndarray, return_xy = False) -> typing.Tuple[int, int]:
        """
		- msk: bool type numpy array
        if return_xy:
            return: (x, y) of opencv coordinate - origin at the upper-left corner
        else:
            return: (raw, col)
		"""
        if (msk == 0).all():
            return None
        coords = Array2D.getCoordGrid2D(msk.shape)
        centroid = coords[msk].mean(axis = 0)
        if return_xy:
            return centroid[::-1]
        else:
            return centroid

    @staticmethod
    def getEdgeMask(msk: np.ndarray, thickness:int = 1, op_for_edge:str = "balance") -> np.ndarray:
        """
        - op_for_edge: dilate / erode / balance
        """
        msk = msk.astype(np.uint8)
        dilate = ndimage.binary_dilation
        erode  = ndimage.binary_erosion
        kernel = np.ones((3,3))
        # dilate = Mask2D._dilate_np
        # erode = Mask2D._dilate_np
        if op_for_edge == "dilate":
            edge_mask = np.logical_xor(dilate(msk, kernel, iterations=thickness), msk)
        elif op_for_edge == "erode":
            edge_mask = np.logical_xor(erode(msk, kernel, iterations=thickness), msk)
        elif op_for_edge == "balance":
            t_erode = thickness//2
            t_dilate = thickness - t_erode
            edge_mask = np.logical_xor(dilate(msk, kernel, iterations=t_dilate), erode(msk, kernel, iterations=t_erode))
        else:
            raise Exception("The op_for_edge keyword can be either dilate / erode / balance")
        return edge_mask

    # def _dilate_np(msk: np.ndarray, kernel = np.ones((3,3)), **kwargs):
        # msk = msk.astype(float)
        # return np.convolve(msk, v = kernel, mode=  "same") > 0

    @staticmethod
    def getDistanceMap(msk: np.ndarray, op_for_edge: str = "dilate") -> np.ndarray:
        """
        - op_for_edge: dilate / erode
        """
        edge_mask = Mask2D.getEdgeMask(msk, thickness=1, op_for_edge=op_for_edge).astype(np.float)
        phi = skfmm.distance(0.5-edge_mask)
        return phi

    @staticmethod
    def getBBoxs(msk: np.ndarray) -> Union[None, List[Tuple[np.ndarray, np.ndarray]]]:
        assert len(msk.shape) == 2, "Mask should be of shape (H, W)"
        if (msk == 0).all():
            return None
        pts = []
        contours, hierarchy = cv.findContours(msk, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = np.squeeze(cnt)
            min_pt = np.min(cnt, axis=0)
            max_pt = np.max(cnt, axis=0)
            pts.append((min_pt, max_pt))
        return pts
    

class MaskEval2D(MaskEvalNd, Mask2D):
    @staticmethod
    def centroidDistance(msk1: np.ndarray, msk2: np.ndarray, spacing = (1,1)) -> float:
        """
        Calculate the centroid distance of the two masks
        """
        assert len(msk1.shape) == 2 and len(msk2.shape) == 2, "Mask should be 2 dimension"
        assert msk1.shape == msk2.shape, "The masks should have same shape"
        c1 = Mask2D.getCentroid(msk1)
        c1 = np.array(c1)
        c2 = Mask2D.getCentroid(msk2)
        c2 = np.array(c2)
        distance = np.linalg.norm((c1-c2)*np.array(spacing))
        return distance
    
    @staticmethod
    def batchCentroidDistance_p(msk1s: np.ndarray, msk2s: np.ndarray, spacing = (1,1), n_workers = -1) -> float:
        def _func(msk_pairs):
            result = np.zeros(len(msk_pairs), dtype=float)
            for i in range(len(msk_pairs)):
                result[i] = MaskEval2D.centroidDistance(*msk_pairs[i], spacing=spacing)
            return result
        return lisJobParallel(_func, list(zip(msk1s, msk2s)), use_buffer=False, n_workers=n_workers)
