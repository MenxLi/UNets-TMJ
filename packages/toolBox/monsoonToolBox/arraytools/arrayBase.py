# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: arrayBase.py                                           | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import numpy as np
from typing import Union, List

class ArrayBase:
	def __init__(self) -> None:
		pass
	@staticmethod
	def stretchArr(arr: np.ndarray, min_val: Union[int, float] = 0, max_val: Union[int, float] = 255, dtype = float)-> np.ndarray:
		if not isinstance(arr, np.ndarray):
			raise Exception("Input should be an ndarray")
		arr = arr.astype(float)
		a = (arr-arr.min())/(arr.max()-arr.min())
		a = a*(max_val-min_val)+min_val
		return a
	@staticmethod
	def mapMatUint8(arr: np.ndarray)->np.ndarray:
		return ArrayBase.stretchArr(arr, 0, 255).astype(np.uint8)

ArrayNd = ArrayBase

class Array2D(ArrayNd):
	@staticmethod
	def imgChannel(img: np.ndarray)->int:
		if len(img.shape)==3:
			return img.shape[2]
		if len(img.shape)==2:
			return 1
	@staticmethod
	def gray2rgb(img: np.ndarray)->np.ndarray:
		new_img = np.concatenate((img[:,:,np.newaxis], img[:,:,np.newaxis], img[:,:,np.newaxis]), axis=2)
		return new_img
	
	def getCoordGrid2D(shape):
		xs = [list(range(shape[1]))]*shape[0]
		xs = np.array(xs)[..., np.newaxis]
		ys = [list(range(shape[0]))]*shape[1]
		ys = np.array(ys).transpose()[..., np.newaxis]
		coords = np.concatenate((ys, xs), axis = -1)
		return coords

class Array3D(ArrayNd):
	@staticmethod
	def getCoordGrid3D(shape):
		a1s = [[list(range(shape[1]))]*shape[0]]*shape[2]
		a1s = np.transpose(np.array(a1s), axes = [1,2,0])[..., np.newaxis]
		a0s = [[list(range(shape[0]))]*shape[2]]*shape[1]
		a0s = np.transpose(np.array(a0s), axes = [2,0,1])[..., np.newaxis]
		a2s = [[list(range(shape[2]))]*shape[1]]*shape[0]
		a2s = np.array(a2s)[...,np.newaxis]

		coords = np.concatenate((a0s,a1s,a2s), axis = -1)
		return coords