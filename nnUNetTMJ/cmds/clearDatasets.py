import os, sys

sys.path.append(os.getcwd())
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnUNetTMJ.utils.pathTools import subDirAndFiles, clearDir

if __name__ == "__main__":
    dir_to_clear = [nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir]
    print("===============Directories to be cleared==================")
    print(dir_to_clear)
    if input("Clear all files under these directories? (y/n): ") == "y":
        for i in dir_to_clear:
            clearDir(i)
    else:
        print("Abort")
