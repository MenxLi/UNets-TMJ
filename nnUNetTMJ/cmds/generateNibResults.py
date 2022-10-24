from monsoonToolBox.env import getEnvVar
import os, shutil


def main():
    infer_output = getEnvVar("infer_output")
    nnunet_raw_db = getEnvVar("nnUNet_raw_data_base")

    folds = [0, 1, 2, 3, 4]
    task = 501
    input_nib_dir = os.path.join(
        nnunet_raw_db, "nnUNet_raw_data/Task501_TMJSeg/imagesTs"
    )
    chk = "model_best"
    tr = "nnUNetTrainerV2"
    for f in folds:
        output_f = os.path.join(infer_output, "f" + str(f))
        if os.path.exists(output_f):
            shutil.rmtree(output_f)
        os.mkdir(output_f)
        code_to_run = "nnUNet_predict -i {input_f} -o {output_f} -t {task} -m 3d_fullres -tr {tr} -chk {chk} --overwrite_existing --save_npz".format(
            input_f=input_nib_dir, output_f=output_f, task=task, tr=tr, chk=chk
        )
        os.system(code_to_run)


if __name__ == "__main__":
    main()
