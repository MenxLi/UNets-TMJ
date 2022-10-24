from monsoonToolBox.statistics.bool1D import Bool1D
from monsoonToolBox.statistics.boolNd import BoolNd
import typing, warnings
import skfmm
import numpy as np 
from typing import Tuple, Union, List, Sequence
from scipy import ndimage
from .arrayBase import ArrayNd
from ..misc import lisJobParallel

NumTVar = typing.TypeVar("NumTVar", int, float)

class MaskNd(ArrayNd):
	@staticmethod
	def area(msk: np.ndarray, area_per_pixel: NumTVar = 1) -> NumTVar:
		return msk.sum()*area_per_pixel

	@staticmethod
	def getEdgeMask(msk: np.ndarray, thickness:int = 1, op_for_edge:str = "balance") -> np.ndarray:
		"""
		- op_for_edge: dilate / erode / balance
		"""
		if op_for_edge == "dilate":
			edge_mask = np.logical_xor(ndimage.binary_dilation(msk, iterations=thickness), msk)
		elif op_for_edge == "erode":
			edge_mask = np.logical_xor(ndimage.binary_erosion(msk, iterations=thickness), msk)
		elif op_for_edge == "balance":
			t_erode = thickness//2
			t_dilate = thickness - t_erode
			edge_mask = np.logical_xor(ndimage.binary_dilation(msk, iterations=t_dilate), ndimage.binary_erosion(msk, iterations=t_erode))
		else:
			raise Exception("The op_for_edge keyword can be either dilate / erode / balance")
		return edge_mask

	@staticmethod
	def getDistanceMap(msk: np.ndarray, op_for_edge: str = "dilate") -> np.ndarray:
		"""
		- op_for_edge: dilate / erode
		"""
		edge_mask = MaskNd.getEdgeMask(msk, thickness=1, op_for_edge=op_for_edge).astype(np.float)
		phi = skfmm.distance(0.5-edge_mask)
		return phi
    
	

class MaskEvalNd(MaskNd):
	#////////////////////////////////////////IoU//////////////////////////////////
	@staticmethod
	def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
		"""
		mask1 and mask2: n dimensional np array of 0 or 1
		"""
		epsilon = 1e-7
		intersection = np.logical_and(mask1, mask2)
		intersection = intersection.sum()
		union = np.logical_or(mask1, mask2)
		union = union.sum() + epsilon
		return intersection/union

	@staticmethod
	def batchIou(mask1s: np.ndarray, mask2s: np.ndarray) -> np.ndarray:
		"""
		calculate ious of a batch of images
		returns 1D array
		"""
		assert mask1s.shape == mask2s.shape, "The masks should have same shape"
		epsilon = 1e-7
		dim = len(mask1s.shape)
		intersection = np.logical_and(mask1s, mask2s)
		union = np.logical_or(mask1s, mask2s)
		for i in range(dim-1):
			intersection = intersection.sum(axis = -1)
			union = union.sum(axis = -1)
		union = union + epsilon
		return intersection/union

	@staticmethod
	def batchIouLoop(mask1s: np.ndarray, mask2s: np.ndarray) -> np.ndarray:
		"""
		calculate ious of a batch of images
		returns 1D array
		"""
		assert len(mask1s) == len(mask2s), "The masks should have same length"
		output = np.array([], dtype=float)
		for m1, m2 in zip(mask1s, mask2s):
			_iou = MaskEvalNd.iou(m1, m2)
			output = np.concatenate((output, [_iou]))
		return output


	#///////////////////////////////////////Dice//////////////////////////////////
	@staticmethod
	def dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
		"""
		mask1 and mask2: n dimensional np array of 0 or 1
		"""
		epsilon = 1e-7
		intersection = np.logical_and(mask1, mask2)
		intersection = intersection.sum()
		denominator = mask1.sum() + mask2.sum() + epsilon
		return 2*intersection/denominator

	@staticmethod
	def batchDice(mask1s: np.ndarray, mask2s: np.ndarray) -> np.ndarray:
		"""
		calculate dices of a batch of images
		returns 1D array
		"""
		assert mask1s.shape == mask2s.shape, "The masks should have same shape"
		epsilon = 1e-7
		dim = len(mask1s.shape)
		intersection = np.logical_and(mask1s, mask2s)
		_mask1s = mask1s.copy()
		_mask2s = mask2s.copy()
		for i in range(dim-1):
			intersection = intersection.sum(axis = -1)
			_mask1s = _mask1s.sum(axis = -1)
			_mask2s = _mask2s.sum(axis = -1)
		denominator = _mask1s + _mask2s + epsilon
		return 2*intersection/denominator
	
	@staticmethod
	def batchDiceLoop(mask1s: np.ndarray, mask2s: np.ndarray) -> np.ndarray:
		"""
		calculate dices of a batch of images
		returns 1D array
		"""
		assert len(mask1s) == len(mask2s), "The masks should have same length"
		output = np.array([], dtype=float)
		for m1, m2 in zip(mask1s, mask2s):
			_dice = MaskEvalNd.dice(m1, m2)
			output = np.concatenate((output, [_dice]))
		return output

	@classmethod
	def sensitivity(cls, gt: np.ndarray, pred: np.ndarray) -> float:
		epsilon = 1e-7
		intersection = np.logical_and(gt, pred)
		intersection = intersection.sum()
		denominator = gt.sum() + epsilon
		return intersection/denominator

	@classmethod
	def batchSensitivity_p(cls, gts: Sequence[np.ndarray], preds: Sequence[np.ndarray]) -> List[float]:
		def _func(mask_pairs):
			out = []
			for pair in mask_pairs:
				out_ = cls.sensitivity(pair[0], pair[1])
				out.append(out_)
			return out
		return lisJobParallel(_func, list(zip(gts, preds)), use_buffer = False)

	@classmethod
	def specificity(cls, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
		epsilon = 1e-7
		intersection = np.logical_and(1-gt, 1-pred)
		intersection = intersection.sum()
		denominator = (1-gt).sum() + epsilon
		return intersection/denominator

	@classmethod
	def batchSpecificity_p(cls, gts: Sequence[np.ndarray], preds: Sequence[np.ndarray]) -> List[float]:
		def _func(mask_pairs):
			out = []
			for pair in mask_pairs:
				out_ = cls.specificity(pair[0], pair[1])
				out.append(out_)
			return out
		return lisJobParallel(_func, list(zip(gts, preds)), use_buffer = False)

	#///////////////////////////////////////////////Hausdorff distance////////////////////////////////////
	@classmethod
	def hausdorffDistance(cls, msk1: np.ndarray, msk2: np.ndarray, spacing:float = 1, is_edge_mask = False) -> Union[float, None]:
		assert msk1.shape == msk2.shape, "The masks should have same shape"
		if (msk1 == 0).all() or (msk2 == 0).all():
			warnings.warn("Zero mask detected when calculating hausdorff distance, return None for this pair.")
			return None
		if not is_edge_mask:
			edge_mask1 = cls.getEdgeMask(msk1, thickness=1, op_for_edge="dilate").astype(np.float)
			edge_mask2 = cls.getEdgeMask(msk2, thickness=1, op_for_edge="dilate").astype(np.float)
		else:
			edge_mask1 = msk1
			edge_mask2 = msk2
		phi1 = skfmm.distance(0.5-edge_mask1)
		phi2 = skfmm.distance(0.5-edge_mask2)
		# flatten valid data
		fv_phi1 = np.ma.masked_array(phi1, mask = 1-edge_mask2).compressed()
		fv_phi2 = np.ma.masked_array(phi2, mask = 1-edge_mask1).compressed()
		dis = np.max((fv_phi1.max(), fv_phi2.max()))
		return dis*spacing

	@classmethod
	def rmsDistance(cls, msk1: np.ndarray, msk2: np.ndarray, spacing: float = 1, is_edge_mask = False) -> Union[float, None]:
		assert msk1.shape == msk2.shape, "The masks should have same shape"
		if (msk1 == 0).all() or (msk2 == 0).all():
			warnings.warn("Zero mask detected when calculating RMS distance, return None for this pair.")
			return None
		if not is_edge_mask:
			edge_mask1 = cls.getEdgeMask(msk1, thickness=1, op_for_edge="dilate").astype(np.float)
			edge_mask2 = cls.getEdgeMask(msk2, thickness=1, op_for_edge="dilate").astype(np.float)
		else:
			edge_mask1 = msk1
			edge_mask2 = msk2
		phi1 = skfmm.distance(0.5-edge_mask1)
		phi2 = skfmm.distance(0.5-edge_mask2)
		# flatten valid data
		fv_phi1 = np.ma.masked_array(phi1, mask = 1-edge_mask2).compressed()
		fv_phi2 = np.ma.masked_array(phi2, mask = 1-edge_mask1).compressed()
		dis1 = np.sqrt((fv_phi1**2).sum())/len(fv_phi1)
		dis2 = np.sqrt((fv_phi2**2).sum())/len(fv_phi2)
		return spacing * (dis1 + dis2)/2

	@classmethod
	def batchRMSDistance(cls, msk1s: typing.Union[np.ndarray, list], msk2s: typing.Union[np.ndarray, list], **kwargs) -> typing.List[Union[float, None]]:
		return [MaskEvalNd.rmsDistance(msk1, msk2, **kwargs) for msk1, msk2 in zip(msk1s, msk2s)]

	@classmethod
	def batchRMSDistance_p(cls, msk1s: typing.Union[np.ndarray, list], msk2s: typing.Union[np.ndarray, list], n_workers = -1, **kwargs) -> typing.List[float]:
		def _func(mask_pairs):
			result = []
			for i in range(len(mask_pairs)):
				result_ = cls.rmsDistance(*mask_pairs[i], **kwargs)
				result.append(result_)
			return result
		return lisJobParallel(_func, list(zip(msk1s, msk2s)), use_buffer=False, n_workers=n_workers)

	@staticmethod
	def batchHausdorffDistance(msk1s: typing.Union[np.ndarray, list], msk2s: typing.Union[np.ndarray, list], **kwargs) -> typing.List[Union[float, None]]:
		return [MaskEvalNd.hausdorffDistance(msk1, msk2, **kwargs) for msk1, msk2 in zip(msk1s, msk2s)]

	@staticmethod
	def batchHausdorffDistance_p(msk1s: typing.Union[np.ndarray, list], msk2s: typing.Union[np.ndarray, list], n_workers = -1, **kwargs) -> typing.List[float]:
		def _func(mask_pairs):
			result = []
			for i in range(len(mask_pairs)):
				result_ = MaskEvalNd.hausdorffDistance(*mask_pairs[i], **kwargs)
				result.append(result_)
			return result
		return lisJobParallel(_func, list(zip(msk1s, msk2s)), use_buffer=False, n_workers=n_workers)

	# ROC curve
	# def getAP(msk1s: np.ndarray, msk2s: np.ndarray, iou_thresh = list(range(0.5,0.95,0.05))) -> List[Tuple[float, float]]:
		# """Unfinished"""
		# ious = MaskEvalNd.batchIouLoop(msk1s, msk2s)
		# # confusions = [MaskEvalNd.__getConfusion(m1, m2) for m1, m2 in zip(msk1s, msk2s)]
		# # tns = [_c["TN"] for _c in confusions]
		# # tps = [_c["TP"] for _c in confusions]
		# # precision = np.array(tns) + np.array(tps)
		# for thresh in iou_thresh:
			# mask = ious>thresh

	def __getConfusion(mask1s: np.ndarray, mask2s: np.ndarray)->dict:
		return BoolNd.calcConfusion(mask1s, mask2s)
