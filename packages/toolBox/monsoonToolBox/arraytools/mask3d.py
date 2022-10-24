import typing
from warnings import resetwarnings
import numpy as np
from .arrayBase import Array3D
from .maskNd import MaskEvalNd, MaskNd
from ..misc import lisJobParallel

class Mask3D(MaskNd, Array3D):
	@staticmethod
	def getCentroid(msk: np.ndarray) -> typing.Tuple[int, int]:
		"""
		- msk: bool type numpy array (3D)
		return: (axis0, axis1, axis2)
		******I'm not sure if the implementation is correct*******
		"""
		assert msk.ndim == 3, "The mask should be 3 dimension, got {}".format(msk.ndim)
		coords = Mask3D.getCoordGrid3D(msk.shape)
		return coords[msk].mean(axis = 0)

class MaskEval3D(MaskEvalNd, Mask3D):
	@staticmethod
	def centroidDistance(msk1: np.ndarray, msk2: np.ndarray, spacing = (1,1,1)) -> float:
		"""
		Calculate the centroid distance of the two masks
		"""
		assert msk1.ndim == 3 and msk2.ndim == 3, "Mask should be 3 dimension"
		assert msk1.shape == msk2.shape, "The masks should have same shape"
		c1 = Mask3D.getCentroid(msk1)
		c1 = np.array(c1)
		c2 = Mask3D.getCentroid(msk2)
		c2 = np.array(c2)
		distance = np.linalg.norm((c1-c2)*np.array(spacing))
		return distance

	@staticmethod
	def batchCentroidDistance_p(msk1s: np.ndarray, msk2s: np.ndarray, spacing = (1,1,1), n_workers = -1) -> typing.List[float]:
		"""
		Calculate the centroid distance of the two masks
		"""
		def _func(msk_pairs):
			result = np.zeros(len(msk_pairs), dtype=float)
			for i in range(len(msk_pairs)):
				result[i] = MaskEval3D.centroidDistance(*msk_pairs[i], spacing=spacing)
			return result
		return lisJobParallel(_func, list(zip(msk1s, msk2s)), use_buffer=False, n_workers=n_workers)