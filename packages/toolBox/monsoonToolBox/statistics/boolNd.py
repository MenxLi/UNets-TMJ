# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: boolNd.py                                              | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #

import pandas as pd
from .basic import StatBasic
import numpy as np
import pprint

class BoolNd(StatBasic):
	@staticmethod
	def _getFormattedPercentage(data: np.ndarray, tag: str = "Percentage"):
		percentage = BoolNd._calcPercentage(data)
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
	def calcConfusion(y_true: np.ndarray, y_pred:np.ndarray) -> dict:
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
	def calcConfusionPrint(y_true: np.ndarray, y_pred:np.ndarray, tag = "Confusion", **pp_kwargs) -> dict:
		"""
		Calculate and print confusion matrix, 
		- y_true and y_pred: 1D array of T/F | 1/0
		"""
		confusion = BoolNd.calcConfusion(y_true, y_pred)
		print("{}:".format(tag))
		# pp = pprint.PrettyPrinter(**pp_kwargs)
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
	def plotConfusion(y_true, y_pred):
		import matplotlib.pyplot as plt
		# https://datatofish.com/confusion-matrix-python/
		import seaborn as sn
		if not isinstance(y_true, np.ndarray):
			y_true = np.array(y_true).astype(np.int)
		if not isinstance(y_pred, np.ndarray):
			y_pred = np.array(y_pred).astype(np.int)
		data = {'y_Actual': y_true,
		'y_Predicted': y_pred }
		df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
		confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

		sn.heatmap(confusion_matrix, annot=True)
		plt.show()

	@staticmethod
	def _convertConfusionMatrixAsDataFrame(confusion):
		data = {"pred_0": pd.Series([confusion["TN"], confusion["FN"]], index = ["true_0", "true_1"]),
				"pred_1": pd.Series([confusion["FP"], confusion["TP"]], index = ["true_0", "true_1"])}
		df = pd.DataFrame(data)
		return df
