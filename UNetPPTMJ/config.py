import os

TRAIN_PATH = os.getenv("TRAIN_PATH")
TEST_PATH = os.getenv("TEST_PATH")
# VALIDATE_PATH = os.getenv("VALIDATE_PATH")

curr_dir = os.path.dirname(__file__)

LBL_NUM = {"Disc": 1, "Condyle": 2, "Eminence": 3}
NUM_LBL = {0: "Background", 1: "Disc", 2: "Condyle", 3: "Eminence"}

# TEMP_DIR = os.path.abspath(".TempDir")
TEMP_DIR = os.path.join(curr_dir, ".TempDir")
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
    print("Created directory: ", TEMP_DIR)
