import os, sys, argparse, shutil, pickle, json
from pprint import pprint
from multiprocessing import Process
from typing import List, Tuple, Dict, Sequence
from monsoonToolBox.arraytools.mask3d import Mask3D
import numpy as np
from .postprocessMasks import postProc
from .config import LBL_NUM, TEMP_DIR
# sys.path.append("/home/monsoon/Documents/Code/toolBox")
import monsoonToolBox as tbx
from monsoonToolBox.arraytools import MaskEval2D, MaskEval3D, Drawer2D, Mask2D
from monsoonToolBox.statistics import Num1D, Bool1D, Stat1D
from monsoonToolBox.logtools import logFuncOutput
from monsoonToolBox.misc import lisJobParallel

import matplotlib.pyplot as plt

# from .machineCheck import JSON_LABEL_IDX_PTH, JSON_LABEL_MACHINE_PTH

def split3DImg(img):
	"""Split the 3D images in to half, seperate the 2 TMJs"""
	sp_id = len(img)//2
	return np.array(img[:sp_id]), np.array(img[sp_id:])

def split4DImg(imgs, deprive_ends = True):
	"""Split many 3D images in to half for each, seperate the 2 TMJs"""
	ims = []
	for i in imgs:
		for j in split3DImg(i):
			ims.append(j)
	return ims

def generateCompareImages(imgs, masks, labels):
	# Generate compare images
	compare_im_dir = os.path.join(TEMP_DIR, "comapreImg")
	if not os.path.exists(compare_im_dir):
		os.mkdir(compare_im_dir)
	color_dict = {
		1: (255, 0, 0),
		2: (0, 255, 0),
		3: (0, 100, 255)
	}
	print("Generating compare images...")
	dir_count = 0
	for ims, msks, lbls in zip(imgs, masks, labels):
		p_dir = os.path.join(compare_im_dir, str(dir_count))
		if os.path.exists(p_dir):
			shutil.rmtree(p_dir)
		os.mkdir(p_dir)
		im_count = 0
		for im, msk, lbl in zip(ims, msks, lbls):
			comp_im = Drawer2D.visualCompareSegmentations(im, [lbl, msk], color_dict=color_dict, alpha=0.5, tags=["Ground truth", "Model prediction"])
			im_path = os.path.join(compare_im_dir, str(dir_count), str(im_count)+".png")
			plt.imsave(im_path, comp_im)
			im_count += 1
		dir_count += 1
	print("Finished generating compare images, the images were saved to ", compare_im_dir)

def batchPostProc_p(lbls, msks) -> List[Tuple[np.ndarray, np.ndarray]]:
	if not isinstance(msks, np.ndarray):
		msks = np.array(msks)
	def _func(msk_pairs):
		m1_results = np.zeros(msks.shape, int)
		m2_results = np.zeros(msks.shape, int)
		for i in range(len(msk_pairs)):
			m1, m2 = postProc(*msk_pairs[i])
			m1_results[i] = m1
			m2_results[i] = m2
		return list(zip(m1_results, m2_results))
	return lisJobParallel(_func, [(m1, m2) for m1, m2 in zip(lbls, msks)], use_buffer=True)

def exportRaw(data: dict, f_path: str):
	with open(f_path, "w") as fp:
		pickle.dump(data, f_path)

def areaCompare(area1: List[int], area2: List[int]):
	"""
	area1: gt
	"""
	assert len(area1) == len(area2)
	ignore_idx: List[int] = []
	for i in range(len(area1)):
		if area1[i] == 0 or area2[i] ==0:
			ignore_idx.append(i)
	for i in ignore_idx[::-1]:
		area1.pop(i)
		area2.pop(i)
	
	area1 = np.array(area1)
	area2 = np.array(area2)
	
	abs_diff = np.abs(area1 - area2)
	rel_diff = abs_diff/area1
	return {
		"abs_diff": abs_diff,
		"rel_diff": rel_diff
	}

def filterDataByIndex(data: Sequence, indexes: List[int]):
	if not indexes:
		return data
	return [data[i] for i in range(len(data)) if i in indexes]

@logFuncOutput("eval_log.txt", mode = "a")
@tbx.misc.timeUtils.timedFunc()
def main():
	# classify according to machine
	MACHINE_CLASSIFY = False	# Not do it for now
	if MACHINE_CLASSIFY:
		MACHINE_LABELS: Dict[str, List[str]] = {}
		MACHINE_INDEX: Dict[str, List[int]] = {}
		with open(JSON_LABEL_MACHINE_PTH, "r") as fp:
			LABEL_MACHINE: Dict[str, str] = json.load(fp)
		with open(JSON_LABEL_IDX_PTH, "r") as fp:
			LABEL_INDEX: Dict[str, int] = json.load(fp)
		for k, v in LABEL_MACHINE.items():
			MACHINE_LABELS.setdefault(v, []).append(k)
		print("Avaliable machines: ", MACHINE_LABELS.keys())
		for machine, label_paths in MACHINE_LABELS.items():
			MACHINE_INDEX[machine] = [LABEL_INDEX[p] for p in label_paths if p in LABEL_INDEX]

		#  _machine_name = list(MACHINE_LABELS.keys())[2]
		_machine_name = None
		if _machine_name is None:
			OUTPUT_DIR = TEMP_DIR
			filter_index = list(range(200))
		else:
			OUTPUT_DIR = os.path.join(TEMP_DIR, _machine_name)
			filter_index = MACHINE_INDEX[_machine_name]
		if not os.path.exists(OUTPUT_DIR):
			os.mkdir(OUTPUT_DIR)
		if filter_index == []:
			exit()

		if _machine_name:
			print(f"Using filter for {_machine_name}: ")
			print(filter_index)
			print("Please evaluate 40 patients npz")
	else:
		OUTPUT_DIR = TEMP_DIR
		filter_index = []
		
	
	npz_path = os.path.join(TEMP_DIR, "results.npz")
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", default=npz_path)
	parser.add_argument("-cim", "--compare_img", action="store_true", default=False)
	args = parser.parse_args()

	process_lis = []
	npz_path = args.file

	data = np.load(npz_path, allow_pickle=True)
	imgs = imgs_ = data["imgs"]
	imgs = filterDataByIndex(imgs, filter_index)
	masks_ = data["masks"]
	masks_ = filterDataByIndex(masks_, filter_index)
	labels_ = data["labels"]
	labels_ = filterDataByIndex(labels_, filter_index)
	if args.compare_img:
		p_cim = Process(target=generateCompareImages, args = (imgs_, masks_, labels_))
		p_cim.start()
		process_lis.append(p_cim)
		# generateCompareImages(imgs_, masks_, labels_)
	print("splitting images into two parts")
	masks_2 = split4DImg(masks_)
	labels_2 = split4DImg(labels_)

	#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Data Formatting\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
	disc_masks = list()
	disc_labels = list()
	condyle_masks = list()
	condyle_labels = list()
	eminence_masks = list()
	eminence_labels = list()

	disc_masks_2d = list()
	disc_labels_2d = list()
	condyle_masks_2d = list()
	condyle_labels_2d = list()
	eminence_masks_2d = list()
	eminence_labels_2d = list()

	RES_XY = 0.46875
	RES_Z = 3.

	print("Postprocessing...")
	for mask_raw, label_raw in zip(masks_2, labels_2):
		mask = list()
		label = list()
		for msk, lbl in zip(mask_raw, label_raw):
			msk, lbl = postProc(msk, lbl)
			mask.append(msk)
			label.append(lbl)
			# 2D data wil be converted into 4D np array
			disc_masks_2d.append(msk == LBL_NUM["Disc"])
			disc_labels_2d.append(lbl == LBL_NUM["Disc"])
			condyle_masks_2d.append(msk == LBL_NUM["Condyle"])
			condyle_labels_2d.append(lbl == LBL_NUM["Condyle"])
			eminence_masks_2d.append(msk == LBL_NUM["Eminence"])
			eminence_labels_2d.append(lbl == LBL_NUM["Eminence"])
		mask = np.array(mask)
		label = np.array(label)

		# 3D data is a list of 3D np array, as different data have different length
		disc_masks.append(mask == LBL_NUM["Disc"])
		disc_labels.append(label == LBL_NUM["Disc"])
		condyle_masks.append(mask == LBL_NUM["Condyle"])
		condyle_labels.append(label == LBL_NUM["Condyle"])
		eminence_masks.append(mask == LBL_NUM["Eminence"])
		eminence_labels.append(label == LBL_NUM["Eminence"])

	# Convert 2D data into np array
	disc_masks_2d = np.array(disc_masks_2d) 
	disc_labels_2d = np.array(disc_labels_2d) 
	condyle_masks_2d = np.array(condyle_masks_2d) 
	condyle_labels_2d = np.array(condyle_labels_2d) 
	eminence_masks_2d = np.array(eminence_masks_2d) 
	eminence_labels_2d = np.array(eminence_labels_2d) 

	image_count_2d = len(disc_masks_2d)
	image_count_3d = len(disc_masks)

	#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\Evaluation\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\	
	#########################3D
	print("Evaluating...", npz_path)
	print("\n>>>>>>>>>>>>>>>>Calculating 3D eval...")
	print("image count: ", image_count_3d)
	disc_dice_3d = MaskEval3D.batchDiceLoop(disc_masks, disc_labels)
	print(Num1D.getFormattedMeanStd(disc_dice_3d, tag = "Disc.Dice - 3D"))
	disc_hd_3d = MaskEval3D.batchHausdorffDistance_p(disc_labels, disc_masks, spacing = RES_XY)
	disc_hd_3d = Bool1D.washNone(disc_hd_3d)[0]
	print(Num1D.getFormattedMeanStd(disc_hd_3d, tag = "Disc.Hausdorff - 3D (unstable)"))
	disc_rms_3d = MaskEval3D.batchRMSDistance_p(disc_labels, disc_masks, spacing = RES_XY)
	disc_rms_3d = Bool1D.washNone(disc_rms_3d)[0]
	print(Num1D.getFormattedMeanStd(disc_rms_3d, tag = "Disc.RMS - 3D"))
	disc_cd_3d = MaskEval3D.batchCentroidDistance_p(disc_labels, disc_masks, spacing = (RES_Z, RES_XY, RES_XY))
	print(Num1D.getFormattedMeanStd(disc_cd_3d, tag = "Disc.Centroid_distance - 3D"))
	disc_area_3d_lbl_raw = [Mask3D.area(m)*(RES_Z * RES_XY**2) for m in disc_labels]
	disc_area_3d_msk_raw = [Mask3D.area(m)*(RES_Z * RES_XY**2) for m in disc_masks]
	area_compare = areaCompare(disc_area_3d_lbl_raw, disc_area_3d_msk_raw)
	print(Num1D.getFormattedMeanStd(area_compare["abs_diff"], tag = "Disc.area_abs_diff - 3D"))
	print(Num1D.getFormattedMeanStd(area_compare["rel_diff"], tag = "Disc.area_rel_diff - 3D"))
		
	condyle_dice_3d = MaskEval3D.batchDiceLoop(condyle_masks, condyle_labels)
	print(Num1D.getFormattedMeanStd(condyle_dice_3d, tag = "Condyle.Dice - 3D"))
	condyle_hd_3d = MaskEval3D.batchHausdorffDistance_p(condyle_labels, condyle_masks, spacing = RES_XY)
	condyle_hd_3d = Bool1D.washNone(condyle_hd_3d)[0]
	print(Num1D.getFormattedMeanStd(condyle_hd_3d, tag = "Condyle.Hausdorff - 3D (unstable)"))
	condyle_rms_3d = MaskEval3D.batchRMSDistance_p(condyle_labels, condyle_masks, spacing = RES_XY)
	condyle_rms_3d = Bool1D.washNone(condyle_rms_3d)[0]
	print(Num1D.getFormattedMeanStd(condyle_rms_3d, tag = "Condyle.RMS - 3D"))
	condyle_area_3d_lbl_raw = [Mask3D.area(m)*(RES_Z * RES_XY**2) for m in condyle_labels]
	condyle_area_3d_msk_raw = [Mask3D.area(m)*(RES_Z * RES_XY**2) for m in condyle_masks]
	area_compare = areaCompare(condyle_area_3d_lbl_raw, condyle_area_3d_msk_raw)
	print(Num1D.getFormattedMeanStd(area_compare["abs_diff"], tag = "Condyle.area_abs_diff - 3D"))
	print(Num1D.getFormattedMeanStd(area_compare["rel_diff"], tag = "Condyle.area_rel_diff - 3D"))

	eminence_hd_3d = MaskEval3D.batchHausdorffDistance_p(eminence_labels, eminence_masks, spacing = RES_XY, is_edge_mask = True)
	eminence_hd_3d = Bool1D.washNone(eminence_hd_3d)[0]
	print(Num1D.getFormattedMeanStd(eminence_hd_3d, tag = "Eminence.Hausdorff - 3D (unstable)"))
	eminence_rms_3d = MaskEval3D.batchRMSDistance_p(eminence_labels, eminence_masks, spacing = RES_XY)
	eminence_rms_3d = Bool1D.washNone(eminence_rms_3d)[0]
	print(Num1D.getFormattedMeanStd(eminence_rms_3d, tag = "Eminence.RMS - 3D"))

	#########################2D
	print("\n>>>>>>>>>>>>>>>>Calculating 2D eval...")
	print("image count: ", image_count_2d)

	disc_confusion = Bool1D.calcConfusionBinaryPrint(Stat1D.notZeros(disc_labels_2d), Stat1D.notZeros(disc_masks_2d), tag = "Disc.Confusion - 2D", indent = 8)
	print("Disc detection accuracy: ", disc_confusion["TP"] + disc_confusion["TN"])
	disc_recog_table = np.logical_and(Stat1D.notZeros(disc_labels_2d), Stat1D.notZeros(disc_masks_2d))
	disc_labels_2d_valid = np.array([disc_labels_2d[i] for i in range(image_count_2d) if disc_recog_table[i]])
	disc_masks_2d_valid = np.array([disc_masks_2d[i] for i in range(image_count_2d) if disc_recog_table[i]])
	disc_dice_2d = MaskEval2D.batchDice(disc_masks_2d_valid, disc_labels_2d_valid)
	print(Num1D.getFormattedMeanStd(disc_dice_2d, tag = "Disc.Dice - 2D"))
	disc_hd_2d = MaskEval2D.batchHausdorffDistance_p(disc_labels_2d_valid, disc_masks_2d_valid, spacing = RES_XY)
	print(Num1D.getFormattedMeanStd(disc_hd_2d, tag = "Disc.Hausdorff - 2D"))
	disc_rms_2d = MaskEval2D.batchRMSDistance_p(disc_labels_2d_valid, disc_masks_2d_valid, spacing = RES_XY)
	print(Num1D.getFormattedMeanStd(disc_rms_2d, tag = "Disc.RMS - 2D"))
	disc_cd_2d = MaskEval2D.batchCentroidDistance_p(disc_labels_2d_valid, disc_masks_2d_valid, spacing = (RES_XY, RES_XY))
	print(Num1D.getFormattedMeanStd(disc_cd_2d, tag = "Disc.Centroid_distance - 2D"))
	disc_area_2d_lbl_raw = [Mask2D.area(m)*(RES_XY**2) for m in disc_labels_2d]
	disc_area_2d_msk_raw = [Mask2D.area(m)*(RES_XY**2)for m in disc_masks_2d]
	area_compare = areaCompare(disc_area_2d_lbl_raw, disc_area_2d_msk_raw)
	print(Num1D.getFormattedMeanStd(area_compare["abs_diff"], tag = "Disc.area_abs_diff - 2D"))
	print(Num1D.getFormattedMeanStd(area_compare["rel_diff"], tag = "Disc.area_rel_diff - 2D"))

	condyle_confusion = Bool1D.calcConfusionBinaryPrint(Stat1D.notZeros(condyle_labels_2d), Stat1D.notZeros(condyle_masks_2d), tag = "Condyle.Confusion - 2D", indent = 8)
	print("Condyle detection accuracy: ", condyle_confusion["TP"] + condyle_confusion["TN"])
	condyle_recog_table = np.logical_and(Stat1D.notZeros(condyle_labels_2d), Stat1D.notZeros(condyle_masks_2d))
	condyle_labels_2d_valid = np.array([condyle_labels_2d[i] for i in range(image_count_2d) if condyle_recog_table[i]])
	condyle_masks_2d_valid = np.array([condyle_masks_2d[i] for i in range(image_count_2d) if condyle_recog_table[i]])
	condyle_dice_2d = MaskEval3D.batchDiceLoop(condyle_masks_2d_valid, condyle_labels_2d_valid)
	print(Num1D.getFormattedMeanStd(condyle_dice_2d, tag = "Condyle.Dice - 2D"))
	condyle_hd_2d = MaskEval3D.batchHausdorffDistance_p(condyle_labels_2d_valid, condyle_masks_2d_valid, spacing = RES_XY)
	print(Num1D.getFormattedMeanStd(condyle_hd_2d, tag = "Condyle.Hausdorff - 2D"))
	condyle_rms_2d = MaskEval2D.batchRMSDistance_p(condyle_labels_2d_valid, condyle_masks_2d_valid, spacing = RES_XY)
	print(Num1D.getFormattedMeanStd(condyle_rms_2d, tag = "Condyle.RMS - 2D"))
	condyle_area_2d_lbl_raw = [Mask2D.area(m)*(RES_XY**2) for m in condyle_labels_2d]
	condyle_area_2d_msk_raw = [Mask2D.area(m)*(RES_XY**2) for m in condyle_masks_2d]
	area_compare = areaCompare(condyle_area_2d_lbl_raw, condyle_area_2d_msk_raw)
	print(Num1D.getFormattedMeanStd(area_compare["abs_diff"], tag = "Condyle.area_abs_diff - 2D"))
	print(Num1D.getFormattedMeanStd(area_compare["rel_diff"], tag = "Condyle.area_rel_diff - 2D"))

	eminence_confusion = Bool1D.calcConfusionBinaryPrint(Stat1D.notZeros(eminence_labels_2d), Stat1D.notZeros(eminence_masks_2d), tag = "Condyle.Confusion - 2D", indent = 8)
	print("Eminence detection accuracy: ", eminence_confusion["TP"] + eminence_confusion["TN"])
	eminence_recog_table = np.logical_and(Stat1D.notZeros(eminence_labels_2d), Stat1D.notZeros(eminence_masks_2d))
	eminence_labels_2d_valid = np.array([eminence_labels_2d[i] for i in range(image_count_2d) if eminence_recog_table[i]])
	eminence_masks_2d_valid = np.array([eminence_masks_2d[i] for i in range(image_count_2d) if eminence_recog_table[i]])
	eminence_hd_2d = MaskEval3D.batchHausdorffDistance_p(eminence_labels_2d_valid, eminence_masks_2d_valid, spacing = RES_XY, is_edge_mask = True)
	print(Num1D.getFormattedMeanStd(eminence_hd_2d, tag = "Eminence.Hausdorff - 2D"))
	eminence_rms_2d = MaskEval2D.batchRMSDistance_p(eminence_labels_2d_valid, eminence_masks_2d_valid, spacing = RES_XY)
	print(Num1D.getFormattedMeanStd(eminence_rms_2d, tag = "Eminence.RMS - 2D"))

	for _p in process_lis:
		_p.join()
	print("Finished.")

	print("Exporting raw results for statistical analysis...")
	exp_data = {
		"2D": {
			"disc":{
				"for_confusion":[Stat1D.notZeros(disc_labels_2d), Stat1D.notZeros(disc_masks_2d)],
				"dice": disc_dice_2d,
				"cd": disc_cd_2d,
				"hd": disc_hd_2d,
				"rms": disc_rms_2d,
			},
			"condyle":{
				"for_confusion":[Stat1D.notZeros(condyle_labels_2d), Stat1D.notZeros(condyle_masks_2d)],
				"dice": condyle_dice_2d,
				"hd": condyle_hd_2d,
				"rms": condyle_rms_2d,
			},
			"eminence":{
				"for_confusion":[Stat1D.notZeros(eminence_labels_2d), Stat1D.notZeros(eminence_masks_2d)],
				"hd": eminence_hd_2d,
				"rms": eminence_rms_2d,
			}
		},
		"3D": {
			"disc":{
				"dice": disc_dice_3d,
				"cd": disc_cd_3d,
				"hd": disc_hd_3d,
				"rms": disc_rms_3d,
			},
			"condyle":{
				"dice": condyle_dice_3d,
				"hd": condyle_hd_3d,
				"rms": condyle_rms_3d,
			},
			"eminence":{
				"hd": eminence_hd_3d,
				"rms": eminence_rms_3d,
			}
		}
	}

	disc_centroid_2d_msk = [MaskEval2D.getCentroid(i) for i in disc_masks_2d_valid]
	disc_centroid_2d_lbl = [MaskEval2D.getCentroid(i) for i in disc_labels_2d_valid]
	disc_centroid_3d_msk = [MaskEval3D.getCentroid(i) for i in disc_masks]
	disc_centroid_3d_lbl = [MaskEval3D.getCentroid(i) for i in disc_labels]
	centroid_data = {
		"2D": {
			"labels": disc_centroid_2d_lbl,
			"masks": disc_centroid_2d_msk
		},
		"3D": {
			"labels": disc_centroid_3d_lbl,
			"masks": disc_centroid_3d_msk
		}
	}

	disc_area_2d_msk = [MaskEval2D.area(i) for i in disc_masks_2d_valid]
	disc_area_2d_lbl = [MaskEval2D.area(i) for i in disc_labels_2d_valid]
	disc_area_3d_msk = [MaskEval3D.area(i) for i in disc_masks]
	disc_area_3d_lbl = [MaskEval3D.area(i) for i in disc_labels]
	area_data = {
		"2D": {
			"labels": disc_area_2d_lbl,
			"masks": disc_area_2d_msk,
			"labels_raw": disc_area_2d_lbl_raw,
			"masks_raw": disc_area_2d_msk_raw,
		},
		"3D": {
			"labels": disc_area_3d_lbl,
			"masks": disc_area_3d_msk,
			"labels_raw": disc_area_3d_lbl_raw,
			"masks_raw": disc_area_3d_msk_raw,
		}
	}

	npz_fname = os.path.basename(npz_path)
	stat_data_output = os.path.join(OUTPUT_DIR, "forStat_"+npz_fname[:-4] + ".pkl")
	centroid_data_output = os.path.join(OUTPUT_DIR, "forCentroid_"+npz_fname[:-4] + ".pkl")
	area_data_output = os.path.join(OUTPUT_DIR, "forArea_"+npz_fname[:-4] + ".pkl")
	with open(stat_data_output, "wb") as fp:
		pickle.dump(exp_data, fp)
	with open(centroid_data_output, "wb") as fp:
		pickle.dump(centroid_data, fp)
	with open(area_data_output, "wb") as fp:
		pickle.dump(area_data, fp)
	print("Finished!")
	print("=========================================================")
	print("")

if __name__ == "__main__":
	main()
