from . import LOGS_DIR, _get_sub_log_dir

import os
import pandas as pd
import ruamel.yaml
import datetime
import time

def compare_train_log(sub_dirs):
    from .Trainer import Trainer
    root_dir = _get_sub_log_dir(Trainer)
    yaml_names = ["setup.yaml", "config.yaml"]
    for yaml_name in yaml_names:
        compare_yaml_files(root_dir, sub_dirs, yaml_name)

def compare_yaml_files(root_dir, sub_dirs, yaml_name):
    data = {}
    all_keys = set()

    for subdirectory in sub_dirs:
        setup_path = os.path.join(root_dir, subdirectory, yaml_name)

        yaml_data = load_yaml(setup_path)

        data[subdirectory] = yaml_data
        all_keys.update(yaml_data.keys())

    diff_data = {}
    for key in all_keys:
        values = {}
        for subdir, subdir_data in data.items():
            value = subdir_data.get(key)
            values[subdir] = value
        value_list = list(values.values())
        try:
            set_ = set(value_list)
        except TypeError:
            set_ = set([str(x) for x in value_list])
        if len(set_) > 1:
            diff_data[key] = values

    diff_df = pd.DataFrame(diff_data)
    print(yaml_name)
    print(diff_df)
    print()


def load_yaml(path) -> dict:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    yaml = ruamel.yaml.YAML()
    with open(path, 'r') as file:
        yaml_data = yaml.load(file)
    return yaml_data

def dump_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml = ruamel.yaml.YAML()
        yaml.dump(data, file)

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

