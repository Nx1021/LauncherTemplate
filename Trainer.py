from . import SCRIPT_DIR, WEIGHTS_DIR, load_yaml
from .BaseLauncher import BaseLogger, Launcher

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

class TrainFlowKW(TypedDict):
    lr_func: str
    lr: float
    cfg: dict

class TrainFlow():
    '''
    训练流程
    '''
    def __init__(self, trainer:"Trainer", flowfile:str|dict[int, TrainFlowKW]) -> None:
        self.trainer = trainer
        self.epoch = 0
        if isinstance(flowfile, dict):
            self.flow:dict[int, TrainFlowKW] = flowfile
        else:
            assert os.path.exists(flowfile), f"{flowfile} not exists!"
            self.flow:dict[int, TrainFlowKW] = load_yaml(flowfile)
        self.stage_segment = list(self.flow.keys())
        if self.stage_segment[0] != 0:
            raise ValueError("the first stage must start at epoch 0!")
        self.scheduler = None

    @property
    def cur_stage(self):
        return sum([self.epoch >= x for x in self.stage_segment]) - 1

    @property
    def total_epoch(self):
        return list(self.flow.keys())[-1]

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

class LossModule(nn.Module):
    """
    method `run` and `get_batch_size` must be overrided
    """
    def __init__(self, loss_names:list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_mngr = LossManager(f"Loss-{self.__class__.__name__}", loss_names)

    def run(self, *args, **kwargs) -> tuple[Tensor]:
        """
        Return a tuple of losses, each Tensor in the tuple is the average of a specific loss item
        """
        pass

    def get_batch_size(self, *args, **kwargs) -> int:
        pass

    def forward(self, *args, **kwargs):
        B = self.get_batch_size(*args, **kwargs)
        losses = self.run(*args, **kwargs)
        assert len(losses) == len(self.loss_mngr.loss_names), f"losses:{len(losses)} != loss_names:{len(self.loss_mngr.loss_names)}"
        self.loss_mngr.record(losses, B)

        return torch.sum(torch.stack(losses))

class LossKW(TypedDict):
    LOSS    = "Loss"

class LossManager():
    '''
    记录总损失
    添加每一个batch的损失
    '''
    def __init__(self, name:str, loss_names:tuple[str]) -> None:
        self.name = name
        self.loss_sum:Tensor = torch.Tensor([0.0])
        self.loss_record:dict[float] = {}
        self.loss_names = loss_names
        self.loss_record[LossKW.LOSS] = 0.0
        self._total_num:int = 0
        self._last_loss = 0.0

    @property
    def loss(self):
        return self.loss_record[LossKW.LOSS]
    
    @property
    def last_loss(self):
        return self._last_loss

    def record(self, losses:tuple[Tensor], num:int):
        self._last_loss = torch.sum(torch.stack(losses)).item()
        for key, value in zip(self.loss_names, losses):
            self.loss_record.setdefault(key, 0.0)            
            new_losss = (self.loss_record[key] * self._total_num + value * num) / (self._total_num + num)
            self.loss_record[key] = new_losss
        self._total_num += num

    def clear(self):
        self.loss_sum[:] = 0
        for k in self.loss_record:
            self.loss_record[k] = 0.0
        self._total_num = 0

class TrainerConfig(TypedDict):
    device:str = "cuda"

    batch_size:int = 8

    start_epoch:int = 0
    train_flow:dict[int, TrainFlowKW] = {}
    distributed:bool = False

    # saving
    saving_period:int = 400

_TEST_TRAINER_ = False

class Trainer(Launcher, abc.ABC):
    def __init__(self,
                 model:nn.Module,
                 train_dataset,
                 val_dataset,
                 criterion: LossModule,
                 config:TrainerConfig):
        super().__init__(model, config["batch_size"])
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        self.flow = TrainFlow(self, config["train_flow"])
        self.distributed = config["distributed"]

        if self.distributed:
            # 初始化分布式环境
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=torch.cuda.device_count(), rank=torch.cuda.current_device())
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.device = torch.device(config["device"])
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())  # 初始化优化器
        self.criterion:LossModule = criterion

        self.best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

        # 创建TensorBoard的SummaryWriter对象，指定保存日志文件的目录
        if not _TEST_TRAINER_:
            self.logger = TrainLogger(self.log_dir)

        self.cur_epoch = 0
        self.start_epoch = config["start_epoch"]
        self.saving_period = config["saving_period"]

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
    def run_model(self, datas:list) -> Union[None, Tensor]:
        pass

    def backward(self, loss:Tensor):
        loss.backward()

    def save_model_checkpoints(self, save_dir, timestamp):
        '''
        保存模型权重
        '''
        if self.distributed:
            torch.save(self.model.module.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))

    def forward_one_epoch(self, dataloader:DataLoader, backward = False):
        '''
        前向传播一个epoch
        '''
        desc = "Train" if backward else "Val"
        if self.skip:
            dataloader = range(len(dataloader))
        progress = tqdm(dataloader, desc=desc, leave=True)
        for datas in progress:
            if not self.skip:
                loss = self.run_model(datas)
                if loss is None:
                    continue
                if torch.isnan(loss).item():
                    print("loss nan, break!")
                    self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp)
                    sys.exit()
                # 反向传播和优化
                self.optimizer.zero_grad()
                if backward and isinstance(loss, torch.Tensor) and loss.grad_fn is not None:
                    self.backward(loss)
            
            if backward:
                self.optimizer.step()

            # 更新进度条信息
            progress.set_postfix({'Loss': "{:>8.4f}".format(self.criterion.loss_mngr.last_loss), "Lr": "{:>2.7f}".format(self.optimizer.param_groups[0]["lr"])})
            if TESTFLOW:
                break
            
        # 将val_loss写入TensorBoard日志文件
        if backward:
            self.logger.write_epoch("Learning rate", self.optimizer.param_groups[0]["lr"], self.cur_epoch)
        for key, value in self.criterion.loss_mngr.loss_record.items():
            self.logger.write_epoch(key, value, self.cur_epoch)

        return self.criterion.loss_mngr.loss()

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
            if epoch % self.saving_period == 0:
                self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp + f"_{self.cur_epoch}_")

            # 更新进度条信息
            tqdm.write('Epoch {} - Train Loss: {:.4f} - Val Loss: {:.4f}'.format(self.cur_epoch, train_loss, val_loss))

        # 保存TensorBoard日志文件
        if _TEST_TRAINER_:
            self.logger.writer.flush()
            self.logger.writer.close()
        if self.distributed:
            dist.destroy_process_group()
