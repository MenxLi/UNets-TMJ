import argparse

from numpy import deg2rad
from ..config import TEMP_DIR
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres, default is 3d_fullres", default="3d_fullres")
	parser.add_argument("-chk", "--checkpoint", help="checkpoint name, model_final_checkpoint / model_latest / model_best, default is model_latest", default="model_latest")
	parser.add_argument("-tr", "--trainer_class_name", help="trainer class name, default: TrainerCustomV2", default="TrainerCustomV2")
	parser.add_argument("-f", "--fold", help = "folds to use for prediction", default="None")

	args = parser.parse_args()

	code_to_run = "nnUNet_predict -i $test_input_TMJ -o $infer_output -t $tmj_task_name \
		-m {model} -chk {checkpoint} -tr {tr} -f {fold} --overwrite_existing --save_npz"\
		.format(model = args.model, checkpoint = args.checkpoint, tr = args.trainer_class_name, fold = args.fold)
	os.system(code_to_run)
	os.system("xdg-open $infer_output")
	

if __name__=="__main__":
	main()