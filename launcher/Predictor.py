import datetime
import os
import pickle
from typing import Generator, Callable, Iterable, Union

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .BaseLauncher import Launcher, BaseLogger

from utils.yaml import load_yaml


class IntermediateManager:
    '''
    中间输出管理器
    用于将中间输出在硬盘上读写
    每次保存一个对象，会在指定的目录下保存一个.pkl文件，并计数。
    '''
    def __init__(self, root_dir, sub_dir_name=""):
        self.root_dir = root_dir
        if sub_dir_name == "":
            current_time = datetime.datetime.now()
            self.sub_dir_name = current_time.strftime("%Y%m%d%H%M%S")
        else:
            self.sub_dir_name = sub_dir_name
        self.save_dir = os.path.join(self.root_dir, self.sub_dir_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.init_save_count() # 不同类型保存计数

    def init_save_count(self):
        '''
        不同类型的保存数量的计数
        '''
        self.save_count = {}

        classnames = os.listdir(self.save_dir)
        for classname in classnames:
            class_path = os.path.join(self.save_dir, classname)
            if os.path.isdir(class_path):
                file_num = len(os.listdir(class_path))
                self.save_count[classname] = file_num

    def load_objects_generator(self, class_name, method:Callable):
        '''
        逐个加载指定类别的所有对象的生成器方法
        返回一个生成器对象，逐个返回对象
        '''
        def load_objects_generator():
            class_dir = os.path.join(self.save_dir, class_name)

            # 检查指定类别的目录是否存在
            if not os.path.isdir(class_dir):
                return

            length = len(os.listdir(class_dir))
            for i in range(length):
                yield method(class_name, i)

        return load_objects_generator()

    def _save_object(self, class_name, obj, save_func:Callable):
        '''
        保存对象到指定的目录中，并根据save_func指定的方法进行保存
        save_func: 以特定方式存储的函数
        '''
        class_dir = os.path.join(self.save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 生成保存文件名，格式为 "class_name_count"
        count = self.save_count.get(class_name, 0) + 1
        file_name = f"{str(count).rjust(6, '0')}"

        # 使用save_func保存对象
        file_path = os.path.join(class_dir, file_name)
        save_func(file_path, obj)

        # 更新保存计数
        self.save_count[class_name] = count

    def _load_object(self, class_name, index:int, load_func:Callable):
        '''
        加载指定类别的所有对象或特定索引的对象
        如果提供了索引参数，则返回对应索引的对象
        否则，返回所有对象的列表
        load_func: 以特定方式加载对象的函数
        '''
        class_dir = os.path.join(self.save_dir, class_name)

        # 检查指定类别的目录是否存在
        if not os.path.isdir(class_dir):
            return []

        objects = []
        file_names = os.listdir(class_dir)
        if index >= 0 and index < len(file_names):
            file_path = os.path.join(class_dir, file_names[index])
            obj = load_func(file_path)
            objects.append(obj)
        else:
            raise IndexError

        return objects

    def save_image(self, class_name, image):
        '''
        保存图像到指定的目录中，并计数
        '''
        def save_image_func(file_path, image):
            file_path += ".png"
            cv2.imwrite(file_path, image)

        self._save_object(class_name, image, save_image_func)

    def load_image(self, class_name, index:int):
        '''
        加载指定类别的图像对象
        '''
        def load_image_func(file_path):
            return cv2.imread(file_path)

        return self._load_object(class_name, index, load_image_func)

    def save_pkl(self, class_name, obj):
        '''
        保存对象为.pkl文件到指定的目录中，并计数
        '''
        def save_pkl_func(file_path, obj):
            file_path += ".pkl"
            with open(file_path, "wb") as file:
                pickle.dump(obj, file)

        self._save_object(class_name, obj, save_pkl_func)

    def load_pkl(self, class_name, index:int):
        '''
        加载指定类别的.pkl文件为对象
        '''
        def load_pkl_func(file_path):
            with open(file_path, "rb") as file:
                return pickle.load(file)

        return self._load_object(class_name, index, load_pkl_func)

class Predictor(Launcher):
    def __init__(self, model, cfg_file,
                  batch_size=32,   
                  log_remark = ""):
        super().__init__(model, batch_size, log_remark)
        self.logger = BaseLogger(self.log_dir)
        self.logger.log(cfg_file)

    @Launcher.timing(-1)
    def example(self):
        pass
