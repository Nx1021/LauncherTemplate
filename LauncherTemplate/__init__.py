import os
import sys

import subprocess
import torch

_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_TEMPLATE_DIR)
SCRIPT_DIR = os.getcwd()

CFG_DIR         = f"{SCRIPT_DIR}/cfg"
DATASETS_DIR    = f"{SCRIPT_DIR}/datasets"
LOGS_DIR        = f"{SCRIPT_DIR}/logs"
WEIGHTS_DIR     = f"{SCRIPT_DIR}/weights"

os.makedirs(CFG_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, '_temp'), exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def _get_sub_log_dir(type):
    return f"{LOGS_DIR}/{type.__name__}_logs/"

def clear_train_logs(epoch_lower_than = 1):
    import shutil
    log_dirs = [x for x in os.listdir(LOGS_DIR) if x.endswith("_logs")]
    for dir_ in log_dirs:
        for d in os.listdir(os.path.join(LOGS_DIR, dir_)):
            if os.path.isdir(os.path.join(LOGS_DIR, dir_, d)) is False:
                continue
            if os.path.exists(os.path.join(LOGS_DIR, dir_, d, "Trainer")) is False:
                continue
            lr_file = os.path.join(LOGS_DIR, dir_, d, "Learning rate.txt")
            rm_ = True
            if os.path.exists(lr_file):
                with open(lr_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > epoch_lower_than:
                        rm_ = False
            if rm_:
                shutil.rmtree(os.path.join(LOGS_DIR, dir_, d))
                continue

def get_gpu_memory(verbose=True) -> dict[tuple[int, int]]:
    command = "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,nounits,noheader"
    output = subprocess.check_output(command, shell=True).decode()
    lines = output.strip().split("\n")

    gpu_memory = {}
    
    for line in lines:
        index, name, used_memory, total_memory = line.split(",")
        used_memory = int(used_memory)
        total_memory = int(total_memory)

        gpu_memory[int(index)] = (used_memory, total_memory)
        
        if verbose:
            memory_info = f"{used_memory}/{total_memory} MB"
            
            gpu_info = f"GPU {index}: {name}, Memory Usage: {memory_info}"
            print(gpu_info)
    
    return gpu_memory

def get_gpu_with_lowest_memory_usage(verbose=True):
    gpu_memory = get_gpu_memory(verbose)
    
    lowest_memory_usage = float('inf')
    gpu_with_lowest_memory = None
    for k, v in gpu_memory.items():
        used_memory, total_memory = v
        if used_memory < lowest_memory_usage:
            lowest_memory_usage = used_memory
            gpu_with_lowest_memory = k
    return gpu_with_lowest_memory
    
def suppress_output():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def enable_output():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# pid     = os.getpid()
# ppid    = os.getppid()
# pid_temp_path = os.path.join(LOGS_DIR, '_temp', "{}.log".format(pid))
# ppid_temp_path = os.path.join(LOGS_DIR, '_temp', "{}.log".format(ppid))
# if os.path.exists(ppid_temp_path):   # 初次运行时，ppid是调试进程，pid是主进程；再次运行时，ppid是主进程，pid是子进程；如果ppid的文件存在，说明是再次运行
#     # print("suppress output")
#     suppress_output()
# else:
#     open(pid_temp_path, "w").close() # 初次运行，pid是主进程，创建一个文件
# clear_train_logs()

min_memory_idx = get_gpu_with_lowest_memory_usage()
print("SCRIPT_DIR:", SCRIPT_DIR)
device = torch.device(f"cuda:{min_memory_idx}")
torch.cuda.set_device(device)
print(f"default GPU idx: {torch.cuda.current_device()}")


from .utils import load_yaml, dump_yaml, wait_until
from .Predictor import Predictor
from .Trainer import Trainer, TrainFlow, TrainFlowKW, TrainerConfig, LossKW, LossManager, _LossManager


# save a example config file
trainer_config_dict:TrainerConfig = {
    "device": "cuda",
    "batch_size": 4,
    "num_workers": 2,
    "shuffle_training": True,
    "persistent_workers": True,
    "empty_cache": True,
    "start_epoch": 1,
    "train_flow": {
        0: {
            "lr_func": "warmup",
            "lr": 1e-6,
            "cfg": {}
        },
        5: {
            "lr_func": "constant",
            "lr": 1e-6,
            "cfg": {}
        },
        30: {
            "lr_func": "cosine",
            "lr": 1e-6,
            "cfg": {}
        },
    },
    "distributed": False,
    "saving_period": 10
}  
if os.path.exists(os.path.join(CFG_DIR, "trainer_config.yaml")) is False and len(os.listdir(CFG_DIR)) == 0:
    dump_yaml(os.path.join(CFG_DIR, "trainer_config.yaml"), trainer_config_dict)