PWD=$(pwd)

# These has to be set accordingly
# export TRAIN_PATH="$HOME/Documents/Data/TMJ-ML/Data_formatted_2/train"
# export TEST_PATH="$HOME/Documents/Data/TMJ-ML/Data_formatted_2/test"
export TRAIN_PATH="./data-demo/train"
export TEST_PATH="./data-demo/test"

# The following can be kept intact
export nnUNet_raw_data_base="$PWD/Database/nnUNet_raw_data_base"
export nnUNet_preprocessed="$PWD/Database/nnUNet_preprocessed"
export RESULTS_FOLDER="$PWD/Database/RESULTS_FOLDER"

export tmj_task_name="Task501_TMJSeg"
export TempDir="$PWD/.TempDir"
mkdir -p $TempDir
export infer_output="$TempDir/nnunetInfer"
mkdir -p $infer_output
export test_input_TMJ="$nnUNet_raw_data_base/nnUNet_raw_data/$tmj_task_name/imagesTs"

alias checkNii="python nnUNetTMJ/cmdTools/visualizeNibabelData.py"
alias clearDatasets="python nnUNetTMJ/cmdTools/clearDatasets.py"
alias clearOutputs="rm -f $infer_output/*; echo \"cleared\""
alias generateResult="python -m nnUNetTMJ.evaluation.generateResults_single"
alias pklTools="python -m nnUNetTMJ.utils.pklTools"
alias evalResult="python -m evaluation.evalResult -f"
alias _removeAll="python -m test.delDirs"

