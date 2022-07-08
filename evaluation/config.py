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

LBL_NUM = {
    "Disc":1,
    "Condyle":2,
    "Eminence":3
}
NUM_LBL = {
    0:"Background",
    1:"Disc",
    2:"Condyle",
    3:"Eminence"
}

CURR_PATH = os.path.dirname(__file__)
TEMP_DIR = os.path.join(CURR_PATH, ".TempDir")
if not os.path.exists(TEMP_DIR):
	os.mkdir(TEMP_DIR)