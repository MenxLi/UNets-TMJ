# +==***---------------------------------------------------------***==+ #
# |                                                                   | #
# |  Filename: config.py                                              | #
# |  Copyright (C)  - All Rights Reserved                             | #
# |  The code presented in this file is part of an unpublished paper  | #
# |  Unauthorized copying of this file, via any medium is strictly    | #
# |  prohibited                                                       | #
# |  Proprietary and confidential                                     | #
# |  Written by Mengxun Li <mengxunli@whu.edu.cn>, June 2022          | #
# |                                                                   | #
# +==***---------------------------------------------------------***==+ #
import os

TRAIN_PATH = os.getenv("TRAIN_PATH")
TEST_PATH = os.getenv("TEST_PATH")
VALIDATE_PATH = os.getenv("VALIDATE_PATH")

NNUNET_DATABASE = os.getenv("nnUnet_raw_data_base")
NNUNET_PREPROCESSED = os.getenv("nnUnet_preprocessed")
NNUNET_RESULTS = os.getenv("RESULTS_FOLDER")

LBL_NUM = {"Disc": 1, "Condyle": 2, "Eminence": 3}
NUM_LBL = {0: "Background", 1: "Disc", 2: "Condyle", 3: "Eminence"}

curr_dir = os.path.dirname(__file__)
TEMP_DIR = os.path.join(curr_dir, ".TempDir")
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
    print("Created directory: ", TEMP_DIR)
