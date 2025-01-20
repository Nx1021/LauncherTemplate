import ruamel.yaml
import os
import sys

import subprocess
import torch

import datetime
import time

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

def load_yaml(path='data.yaml') -> dict:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    def load():
        with open(path, 'r') as file:
            yaml = ruamel.yaml.YAML()
            return yaml.load(file)

    yaml_data = load()

    return yaml_data

def dump_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml = ruamel.yaml.YAML()
        yaml.dump(data, file)

def suppress_output():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def enable_output():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def clear_train_logs(epoch_lower_than = 1):
    import shutil
    log_dirs = [x for x in os.listdir(LOGS_DIR) if x.endswith("_logs")]
    for dir_ in log_dirs:
        for d in os.listdir(os.path.join(LOGS_DIR, dir_)):
            if os.path.isdir(os.path.join(LOGS_DIR, dir_, d)) is False:
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
# clear_train_logs()

def wait_until(target_time):
    """
    阻塞程序直到指定时间到达，每秒打印剩余时间。

    参数:
        target_time (str): 目标时间的字符串，格式为 'YYYY-MM-DD HH:MM:SS'。
    """
    # 解析输入的目标时间
    target_time = datetime.datetime.strptime(target_time, '%Y-%m-%d %H:%M:%S')
    check_interval = 60
    _state = 0
    while True:
        # 获取当前时间
        now = datetime.datetime.now()
        
        # 计算剩余时间
        remaining_time = target_time - now

        # 如果到达或超过目标时间，退出循环
        remain_seconds = remaining_time.total_seconds()
        if remain_seconds <= 0:
            print("Target time reached!")
            break
        if remain_seconds < 600:
            check_interval = 0.5
            _state = 1

        # 计算剩余的小时、分钟和秒
        hours, remainder = divmod(remaining_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 打印剩余时间
        print_string = f"Plan to start at {target_time.strftime('%Y-%m-%d %H:%M:%S')}"
        if _state == 0:
            print(f"\r {print_string}. {hours} h {minutes} min remains         ", end = '')
        elif _state == 1:
            print(f"\r {print_string}. {minutes} min {seconds} sec remains         ", end = '')

        # 每秒钟检查一次
        time.sleep(check_interval)


pid     = os.getpid()
ppid    = os.getppid()
pid_temp_path = os.path.join(LOGS_DIR, '_temp', "{}.log".format(pid))
ppid_temp_path = os.path.join(LOGS_DIR, '_temp', "{}.log".format(ppid))
if os.path.exists(ppid_temp_path):   # 初次运行时，ppid是调试进程，pid是主进程；再次运行时，ppid是主进程，pid是子进程；如果ppid的文件存在，说明是再次运行
    # print("suppress output")
    suppress_output()
else:
    open(pid_temp_path, "w").close() # 初次运行，pid是主进程，创建一个文件
    clear_train_logs()

min_memory_idx = get_gpu_with_lowest_memory_usage()
print("SCRIPT_DIR:", SCRIPT_DIR)
device = torch.device(f"cuda:{min_memory_idx}")
torch.cuda.set_device(device)
print(f"default GPU idx: {torch.cuda.current_device()}")



from .Predictor import Predictor
from .Trainer import Trainer, TrainFlow, TrainFlowKW, TrainerConfig, LossKW, LossManager, _LossManager
