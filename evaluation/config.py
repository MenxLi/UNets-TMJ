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