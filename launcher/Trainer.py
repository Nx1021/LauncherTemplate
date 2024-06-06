from . import SCRIPT_DIR, WEIGHTS_DIR

import os
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ConstantLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from .BaseLauncher import BaseLogger, Launcher
from utils.yaml import load_yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import platform
import sys

import time
import abc

from typing import Union, TypedDict
sys_name = platform.system()
if sys_name == "Windows":
    TESTFLOW = False
else:
    TESTFLOW = False

class TrainFlow():
    '''
    训练流程
    '''
    def __init__(self, trainer:"Trainer", flowfile) -> None:
        self.trainer = trainer
        self.epoch = 0
        self.flow:dict = load_yaml(flowfile, False)
        self.stage_segment = list(self.flow.keys())
        if self.stage_segment[0] != 0:
            raise ValueError("the first stage must start at epoch 0!")
        self.scheduler = None

    @property
    def cur_stage(self):
        return sum([self.epoch >= x for x in self.stage_segment]) - 1

    def get_lr_func(self, lr_name, totol_step, initial_lr):
        # totol_step = totol_step * int(np.round(len(self.trainer.train_dataset) / self.trainer.batch_size))
        for param_group in self.trainer.optimizer.param_groups:
            param_group['initial_lr'] = initial_lr
            param_group['lr'] = initial_lr
        if lr_name == "warmup":
            return LambdaLR(self.trainer.optimizer, 
                            lr_lambda=lambda step: min(step / totol_step, 1.0))
        if lr_name == "cosine":
            return CosineAnnealingLR(self.trainer.optimizer, 
                                                        totol_step)
        if lr_name == "constant":
            return ConstantLR(self.trainer.optimizer, 1.0, 1)
    
    def enter_new_stage(self):
        stage_info = self.flow[self.stage_segment[self.cur_stage]]
        if stage_info is None:
            return
        totol_step = self.stage_segment[self.cur_stage + 1] - self.stage_segment[self.cur_stage]
        self.scheduler = self.get_lr_func(stage_info["lr_func"], totol_step, stage_info["lr"])
        # self.trainer.optimizer.zero_grad()
        # self.trainer.optimizer.step()
        if "cfg" in stage_info:
            self.trainer.inner_model.cfg.update(stage_info["cfg"])

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.epoch >= self.stage_segment[-1]:
            raise StopIteration      
        if self.epoch in self.stage_segment:
            self.enter_new_stage()    
        if self.scheduler is not None:
            self.scheduler.step()             
        self.epoch += 1 
        return self.epoch

class TrainLogger(BaseLogger):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.writer = SummaryWriter(log_dir)

    def write_epoch(self, tag, value, step):
        if isinstance(value, torch.Tensor):
            value = value.item()
        # 写入 SummaryWriter
        self.writer.add_scalar(tag, value, step)

        # 写入文件
        log_file = os.path.join(self.log_dir, f"{tag}.txt")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = "{:>4}  \t{}  \t{:<10.6f}\n".format(step, current_time, value)

        with open(log_file, 'a') as f:
            f.write(log_line)

class LossKW(TypedDict):
    LOSS    = "Loss"
    
class LossResult():
    def __init__(self, loss_Tensor:Tensor, item_weights:Tensor, item_names:list[str]) -> None:
        '''
        parameter
        -----
        * loss_Tensor: Tensor [B, num_items]
        * item_weights: the weight of each column [num_items]

        len(item_weights) must be equal to loss_Tensor.shape[1]
        '''
        assert len(loss_Tensor.shape) == 2, "loss_Tensor must be 2D"
        assert len(item_weights.shape) == 1 and item_weights.numel() == len(item_names) == loss_Tensor.shape[1] 
        self.loss_Tensor = loss_Tensor
        self.item_weights = item_weights.to(loss_Tensor.device)
        self.item_names = item_names

    def apply_weight(self):
        if self.valid:
            loss_Tensor = self.loss_Tensor * self.item_weights # [B, num_items]
            return loss_Tensor
        else:
            return self.loss_Tensor

    def loss(self) -> Tensor:
        if self.valid:
            loss_Tensor = self.apply_weight() # [B, num_items]
            loss = torch.mean(torch.sum(loss_Tensor, dim=-1)) # [1] 
            return loss
        else:
            return torch.Tensor([0.0]).to(self.loss_Tensor.device) # [1] 

    @property
    def valid(self):
        return self.loss_Tensor.numel() != 0

    @property
    def B(self):
        return self.loss_Tensor.shape[0]

    @property
    def num_items(self):
        return self.loss_Tensor.shape[1]

    def to_dict(self) -> dict:
        dict_ = {}
        if self.valid:
            loss_Tensor:np.ndarray = self.apply_weight().detach().cpu().numpy()
            item_losses = np.mean(loss_Tensor, (0, 1))
            for name, v in zip(self.item_names, item_losses):
                dict_[name] = float(v)
            dict_[LossKW.LOSS]   = float(np.mean(np.sum(loss_Tensor, -1)))
        return dict_

    @staticmethod
    def concat(result_list: list["LossResult"]):
        if len(result_list) > 0:
            item_weights: Tensor    = result_list[0].item_weights
            item_names: list[str]   = result_list[0].item_names

            concat = torch.concat([r.loss_Tensor for r in result_list])

            return LossResult(concat, item_weights, item_names)
        else:
            return LossResult(torch.zeros(0,0,0), None, None)

class LossRecorder():
    def __init__(self, name, top = True) -> None:
        self.name = name
        self.loss_sum:Tensor = torch.Tensor([0.0])
        self.loss_record:dict[float] = {}
        self.loss_record[LossKW.LOSS] = 0.0
        self.detect_num:int = 0
        if top:
            self.buffer = LossRecorder("__buffer", top=False)

    def record(self, result:LossResult):
        for key, value in result.to_dict().items():
            self.buffer.loss_record.setdefault(key, 0.0)
            self.buffer.loss_record[key] += value * result.B
        self.buffer.detect_num += result.B

    def clear(self):
        self.loss_sum = torch.Tensor([0.0])
        self.loss_record.clear()
        self.loss_record[LossKW.LOSS] = 0.0
        self.detect_num = 0

    def loss(self):
        return self.__get_mean(self.loss_record[LossKW.LOSS])

    def __get_mean(self, value:float):
        if self.detect_num == 0:
            return 0.0
        else:
            return value / self.detect_num

    def to_dict(self):
        dict_:dict[str, float] = {}
        for key, sum_value in self.loss_record.items():
            dict_[self.name + ' ' + key] = self.__get_mean(sum_value)

        return dict_

    def merge(self):
        buffer = self.buffer
        self.detect_num += buffer.detect_num
        for key, value in buffer.loss_record.items():
            self.loss_record.setdefault(key, 0.0)
            self.loss_record[key] += value

_TEST_TRAINER_ = False

class Trainer(Launcher, abc.ABC):
    def __init__(self,
                 model:nn.Module,
                 train_dataset,
                 val_dataset,
                 criterion,
                 batch_size,
                 flow_file = "",
                 distribute = False,
                 start_epoch = 0,
                 save_period = 400):
        super().__init__(model, batch_size)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.flow = TrainFlow(self, flow_file)
        self.distribute = distribute

        if self.distribute:
            # 初始化分布式环境
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=torch.cuda.device_count(), rank=torch.cuda.current_device())
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())  # 初始化优化器
        self.criterion:nn.Module = criterion

        self.best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

        # 创建TensorBoard的SummaryWriter对象，指定保存日志文件的目录
        if not _TEST_TRAINER_:
            self.logger = TrainLogger(self.log_dir)

        self.cur_epoch = 0
        self.start_epoch = start_epoch
        self.save_period = save_period

        self.collate_fn = None

    @property
    def skip(self):
        return self.cur_epoch < self.start_epoch

    @property
    def inner_model(self) -> nn.Module:
        if isinstance(self.model, torch.nn.DataParallel):
            module:nn.Module= self.model.module # type: ignore
            return module
        else:
            return self.model

    @abc.abstractmethod
    def run_model(self, datas:list, ldmk_loss_mngr:LossRecorder) -> Union[None, Tensor]:
        pass

    def save_model_checkpoints(self, save_dir, timestamp):
        '''
        保存模型权重
        '''
        if self.distribute:
            torch.save(self.model.module.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))

    def forward_one_epoch(self, dataloader:DataLoader, backward = False):
        '''
        前向传播一个epoch
        '''
        desc = "Train" if backward else "Val"
        ldmk_loss_mngr = LossRecorder(desc)
        if self.skip:
            dataloader = range(len(dataloader))
        progress = tqdm(dataloader, desc=desc, leave=True)
        for datas in progress:
            if not self.skip:
                loss = self.run_model(datas, ldmk_loss_mngr)
                if loss is None:
                    continue
                if torch.isnan(loss).item():
                    print("loss nan, break!")
                    self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp)
                    sys.exit()
                # 反向传播和优化
                self.optimizer.zero_grad()
                if backward and ldmk_loss_mngr.buffer.detect_num > 0 and isinstance(loss, torch.Tensor) and loss.grad_fn is not None:
                    loss.backward()
            
            if backward:
                self.optimizer.step()

            # 更新进度条信息
            progress.set_postfix({'Loss': "{:>8.4f}".format(ldmk_loss_mngr.loss()), "Lr": "{:>2.7f}".format(self.optimizer.param_groups[0]["lr"])})
            if TESTFLOW:
                break
            ldmk_loss_mngr.buffer.clear()
            
        # 将val_loss写入TensorBoard日志文件
        if backward:
            self.logger.write_epoch("Learning rate", self.optimizer.param_groups[0]["lr"], self.cur_epoch)
        for key, value in ldmk_loss_mngr.to_dict().items():
            self.logger.write_epoch(key, value, self.cur_epoch)

        return ldmk_loss_mngr.loss()

    def train_one_epoch(self, dataloader):
        self.inner_model.train()
        return self.forward_one_epoch(dataloader, True)

    def val_one_epoch(self, dataloader):
        self.inner_model.eval()
        with torch.no_grad():
            return self.forward_one_epoch(dataloader, False)

    def train(self):
        print("start to train... time:{}".format(self.start_timestamp))
        self.cur_epoch = 0
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,  collate_fn=self.collate_fn)
        val_dataloader   = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        for epoch in self.flow:
            self.cur_epoch = epoch
            tqdm.write('\nEpoch {} start...'.format(self.cur_epoch))
            # 训练阶段
            train_loss = self.train_one_epoch(train_dataloader)

            # 验证阶段
            val_loss = self.val_one_epoch(val_dataloader)

            # 如果验证损失低于历史最小值，则保存模型权重
            if val_loss < self.best_val_loss and not self.skip:
                print("new best val_loss: {}, saving...".format(val_loss))
                self.best_val_loss = val_loss
                self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp)
            if epoch % self.save_period == 0:
                self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp + f"_{self.cur_epoch}_")

            # 更新进度条信息
            tqdm.write('Epoch {} - Train Loss: {:.4f} - Val Loss: {:.4f}'.format(self.cur_epoch, train_loss, val_loss))

        # 保存TensorBoard日志文件
        if _TEST_TRAINER_:
            self.logger.writer.flush()
            self.logger.writer.close()
        if self.distribute:
            dist.destroy_process_group()
