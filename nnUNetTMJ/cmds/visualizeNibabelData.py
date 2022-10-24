import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
from monsoonToolBox.arraytools import ArrayNd


if __name__ == "__main__":
	example_filename = sys.argv[1]
	img = nib.load(example_filename)
	img.set_data_dtype(float)
	print(img.header['db_name'])  # 输出头信息

	width, height, queue = img.dataobj.shape
	imgs = np.array(img.dataobj)
	imgs = ArrayNd.stretchArr(imgs, 0, 255)
	OrthoSlicer3D(imgs).show()