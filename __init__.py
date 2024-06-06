import sys
import platform
import os
import shutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Linux":
    # 切换工作目录到当前脚本所在的目录
    os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

CFG_DIR         = f"{SCRIPT_DIR}/cfg"
DATASETS_DIR    = f"{SCRIPT_DIR}/datasets"
LOGS_DIR        = f"{SCRIPT_DIR}/logs"
WEIGHTS_DIR     = f"{SCRIPT_DIR}/weights"
SERVER_DATASET_DIR = "/home/nerc-ningxiao/datasets"

def _get_sub_log_dir(type):
    return f"{LOGS_DIR}/{type.__name__}_logs/"
