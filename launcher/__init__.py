import os
import sys

import subprocess
import torch

_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_TEMPLATE_DIR)
SCRIPT_DIR = os.getcwd()
print("SCRIPT_DIR:", SCRIPT_DIR)

CFG_DIR         = f"{SCRIPT_DIR}/cfg"
DATASETS_DIR    = f"{SCRIPT_DIR}/datasets"
LOGS_DIR        = f"{SCRIPT_DIR}/logs"
WEIGHTS_DIR     = f"{SCRIPT_DIR}/weights"

os.makedirs(CFG_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def _get_sub_log_dir(type):
    return f"{LOGS_DIR}/{type.__name__}_logs/"

def get_gpu_with_lowest_memory_usage():
    command = "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,nounits,noheader"
    output = subprocess.check_output(command, shell=True).decode()
    lines = output.strip().split("\n")
    
    lowest_memory_usage = float('inf')
    gpu_with_lowest_memory = None
    
    for line in lines:
        index, name, used_memory, total_memory = line.split(",")
        used_memory = int(used_memory)
        total_memory = int(total_memory)
        
        memory_info = f"{used_memory}/{total_memory} MB"
        
        gpu_info = f"GPU {index}: {name}, Memory Usage: {memory_info}"
        print(gpu_info)
        
        if used_memory < lowest_memory_usage:
            lowest_memory_usage = used_memory
            gpu_with_lowest_memory = int(index)
    
    return gpu_with_lowest_memory

min_memory_idx = get_gpu_with_lowest_memory_usage()
device = torch.device(f"cuda:{min_memory_idx}")
torch.cuda.set_device(device)
print(f"default GPU idx: {torch.cuda.current_device()}")