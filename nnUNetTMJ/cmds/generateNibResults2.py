# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: generateNibResults2.py                                 | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
"""For time test"""
from monsoonToolBox.env import getEnvVar
from monsoonToolBox.logtools import timedFunc
import os,shutil

@timedFunc()
def main():
	infer_output = getEnvVar("infer_output")

	folds = [0]
	task = 501
	input_nib_dir = "/home/monsoon/Documents/Code/TMJ-ML-torch/Database/nnUNet_raw_data_base/nnUNet_raw_data/Task501_TMJSeg/imagesTs"
	chk = "model_best"
	# chk = "model_final_checkpoint"
	tr = "nnUNetTrainerV2"
	for f in folds:
		output_f = os.path.join(infer_output, "f"+str(f))
		if os.path.exists(output_f):
			shutil.rmtree(output_f)
		os.mkdir(output_f)
		code_to_run = "nnUNet_predict -i {input_f} -o {output_f} -t {task} -m 3d_fullres -tr {tr} -chk {chk} --num_threads_preprocessing 8 --overwrite_existing --disable_tta --all_in_gpu None --mode normal".format(
			input_f = input_nib_dir, output_f = output_f, task = task, tr = tr, chk = chk
		)
		os.system(code_to_run)

if __name__ == "__main__":
	main()
