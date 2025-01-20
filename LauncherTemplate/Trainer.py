import torch.utils
import torch.utils.data
from . import SCRIPT_DIR, WEIGHTS_DIR, load_yaml, LOGS_DIR
from .BaseLauncher import BaseLogger, Launcher, create_folder_shortcut, MODEL_TYPE

import os
import win32com.client
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
import datetime
import platform
import sys

import time
import abc

from typing import Union, TypedDict, Callable, Optional, Iterable, Generic, TypeVar
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
    def __init__(self, trainer:"Trainer", flowfile:Union[str,dict[int, TrainFlowKW]]) -> None:
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
        os.makedirs(log_dir, exist_ok=True)
        self._on = True

    def set_logger_on(self, value = True):
        self._on = value

    def write_epoch(self, tag, value, step):
        if not self._on:
            return
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
    
    def flush_log(self):
        if not self._on:
            return
        self.writer.flush()
        self.writer.close()

LOSSMODULE = TypeVar("LOSSMODULE", bound=nn.Module)
class LossManager(Generic[LOSSMODULE]):
    def __init__(self, loss:Optional[Callable] = None, losspart_names:Optional[Iterable[str]] = None) -> None:
        self.loss_module:LOSSMODULE = loss
        self._mode:bool = True
        self.losspart_names = losspart_names 
        if self.losspart_names is not None:
            self.init_loss_mngr(self.losspart_names)
        else:
            self.init_loss_mngr([])

    def init_loss_mngr(self, losspart_names:Iterable[str]):
        self.train_loss_mngr = _LossManager(losspart_names)
        self.val_loss_mngr   = _LossManager(losspart_names)

    @property
    def mode_str(self):
        if self._mode:
            return "train"
        else:
            return "val"

    @property
    def loss_mngr(self):
        if self._mode:
            return self.train_loss_mngr
        else:
            return self.val_loss_mngr
    
    def get_batch_size(self, *args, **kwargs) -> int:
        return args[0].shape[0]

    def __call__(self, *args, BATCHSIZE = None, **kwargs):
        B = self.get_batch_size(*args, **kwargs) if BATCHSIZE is None else BATCHSIZE
        if self.loss_module is None:
            losses = args[0]
        else:
            losses:Union[tuple[Tensor], dict[str, Tensor], Tensor] = self.loss_module(*args, **kwargs)
        with torch.no_grad():
            if isinstance(losses, (tuple, dict)):
                # multiple loss items
                if self.losspart_names is None:
                    self.losspart_names = [f"loss_{i}" for i in range(len(losses))]
                    self.init_loss_mngr(self.losspart_names)
                assert len(losses) == len(self.losspart_names), f"expect {len(self.losspart_names)} loss items, but got {len(losses)} items"
                if isinstance(losses, tuple):
                    losses = {k: v for k, v in zip(self.losspart_names, losses)}
                self.loss_mngr.record(losses, B)
            elif isinstance(losses, Tensor):
                if self.losspart_names is None:
                    self.losspart_names = ["loss"]
                    self.init_loss_mngr(self.losspart_names)
                _len = len(self.losspart_names)
                assert _len == 1, f"expect 1 loss item, but got {_len} items"
                losses = {k: v for k, v in zip(self.losspart_names, [losses])}
                self.loss_mngr.record(losses, B)
            else:
                raise ValueError(f"losses type:{type(losses)} not supported")
        
        if self._mode:
            loss_ = torch.sum(torch.stack([v for k, v in losses.items() if v.grad_fn is not None]))
        else:
            loss_ = torch.sum(torch.stack([v for k, v in losses.items()]))
        return loss_

    def train_mode(self, value = True):
        self._mode:bool = value

    def val_mode(self, value = True):
        self._mode:bool = not value

class LossKW(TypedDict):
    LOSS    = "Loss"

class _LossManager():
    '''
    记录总损失
    添加每一个batch的损失
    '''
    def __init__(self, loss_names:tuple[str]) -> None:
        self.loss_record:dict[str, Tensor] = {}
        self.__loss_names = loss_names
        self.loss_record[LossKW.LOSS] = torch.tensor(0.0)
        for key in loss_names:
            self.loss_record[key] = torch.tensor(0.0)
        self._total_num:int = 0
        self._last_loss = 0.0

    @property
    def loss(self):
        return self.loss_record[LossKW.LOSS]
    
    @property
    def loss_value(self):
        return float(self.loss_record[LossKW.LOSS].item())

    @property
    def last_loss_value(self):
        return self._last_loss

    def record(self, losses:dict[str, Tensor], num:int):
        # TODO: 逻辑没有理清楚，record模块只应该用于记录，不应该计算
        self._last_loss = float(self.loss.item())
        # self._last_loss = torch.sum(torch.stack(losses)).item()
        for key, value in losses.items():
            if key not in self.__loss_names:
                raise ValueError(f"loss name:{key} not in {self.__loss_names}")
            new_losss = (self.loss_record[key] * self._total_num + value * num) / (self._total_num + num)
            self.loss_record[key] = new_losss
        self._total_num += num

        loss_parts = [v for k, v in self.loss_record.items() if k is not LossKW.LOSS]
        self.loss_record[LossKW.LOSS] = torch.sum(torch.stack(loss_parts))

    def clear(self):
        for k in list(self.loss_record.keys()):
            del self.loss_record[k]
            self.loss_record[k] = torch.tensor(0.0)
        self._total_num = 0

class TrainerConfig(TypedDict):
    device:str      = "cuda"

    batch_size:int  = 8
    num_workers:int = 4

    start_epoch:int = 0
    train_flow:dict[int, TrainFlowKW] = {}
    distributed:bool = False

    # saving
    saving_period:int = 400

_TEST_TRAINER_ = False

MODEL_TYPE = TypeVar("MODEL_TYPE", bound=nn.Module)
LOSS_TYPE  = TypeVar("LOSS_TYPE", bound=nn.Module)
class Trainer(Launcher[MODEL_TYPE], abc.ABC, Generic[MODEL_TYPE, LOSS_TYPE]):
    def __init__(self,
                 model:MODEL_TYPE,
                 train_dataset,
                 val_dataset,
                 criterion: Union[Callable, nn.Module, LossManager],
                 config:TrainerConfig):
        super().__init__(model, config)
        self.train_dataset:torch.utils.data.Dataset = train_dataset
        self.val_dataset:torch.utils.data.Dataset   = val_dataset

        self.flow = TrainFlow(self, config["train_flow"])

        self.optimizer = optim.Adam(self.model.parameters())  # 初始化优化器
        if not isinstance(criterion, LossManager):
            criterion = LossManager(criterion)
        self.criterion:LossManager[LOSS_TYPE] = criterion

        self.best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

        # 创建TensorBoard的SummaryWriter对象，指定保存日志文件的目录
        if not _TEST_TRAINER_:
            self.logger = TrainLogger(self.log_dir)

        self.cur_epoch = 0
        self.start_epoch    = config["start_epoch"]
        self.saving_period  = config["saving_period"]
        self._training_forward = True

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
        self.optimizer.step()

    def loss_nan_process(self, loss:Tensor):
        print("loss nan, continue! zero grad")
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return True

    def save_model_checkpoints(self, save_dir, timestamp):
        '''
        保存模型权重
        '''
        if self.distributed:
            torch.save(self.model.module.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))
        short_cut_path = os.path.join(WEIGHTS_DIR, self._log_sub_dir+'_training_logs.lnk')
        if not os.path.exists(short_cut_path):
            create_folder_shortcut(self.log_dir, short_cut_path, "link of logs")

    def forward_one_epoch(self, dataloader:DataLoader, _training_forward = False):
        '''
        前向传播一个epoch
        '''
        self._training_forward = _training_forward
        desc = "Train" if _training_forward else "Val"
        if self.skip:
            dataloader = range(len(dataloader))
        progress = tqdm(dataloader, desc=desc, leave=True)
        for datas in progress:
            if self.skip:
                progress.set_postfix({'SKIP': "True"})
                break
            if not self.skip:
                loss = self.run_model(datas)
                if loss is None:
                    continue
                if torch.isnan(loss).item():
                    print("loss nan, continue! zero grad")
                    if_continue = self.loss_nan_process(loss)
                    if if_continue:
                        continue
                    else:
                        break
                # 反向传播和优化
                self.optimizer.zero_grad()
                if _training_forward and isinstance(loss, torch.Tensor) and loss.grad_fn is not None:
                    self.backward(loss)
            
            # if backward:
            #     self.optimizer.step()

            # 更新进度条信息
            progress.set_postfix({'Loss': "{:>8.4f}".format(self.criterion.loss_mngr.last_loss_value), "Lr": "{:>2.7f}".format(self.optimizer.param_groups[0]["lr"])})
            if TESTFLOW:
                break
            
        # 将val_loss写入TensorBoard日志文件
        if _training_forward:
            self.logger.write_epoch("Learning rate", self.optimizer.param_groups[0]["lr"], self.cur_epoch)
        for key, value in self.criterion.loss_mngr.loss_record.items():
            self.logger.write_epoch(self.criterion.mode_str + '-' + key, value, self.cur_epoch)

        return self.criterion.loss_mngr.loss_value

    def train_one_epoch(self, dataloader):
        self.inner_model.train()
        self.criterion.train_mode(True)
        self.criterion.loss_mngr.clear()
        return self.forward_one_epoch(dataloader, True)

    def val_one_epoch(self, dataloader):
        self.inner_model.eval()
        self.criterion.train_mode(False)
        self.criterion.loss_mngr.clear()
        with torch.no_grad():
            return self.forward_one_epoch(dataloader, False)
    
    def train(self):
        print("start to train... time:{}".format(self.start_timestamp))
        self.cur_epoch = 0
        # num_workers = 0 if sys_name == "Windows" else 4
        num_workers = self.config.get("num_workers", 0)
        persistent_workers = self.config.get("persistent_workers", True)
        persistent_workers = False if num_workers == 0 else persistent_workers
        shuffle_training = self.config.get('shuffle_training', True)
        train_dataloader = self._dataloader_type(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle_training,  
                                      collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)
        val_dataloader   = self._dataloader_type(self.val_dataset,   batch_size=self.batch_size, shuffle=False, 
                                      collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)

        for epoch in self.flow:
            self.cur_epoch = epoch
            start_time = time.time()
            tqdm.write('\nEpoch {} start... time: {}'.format(self.cur_epoch, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
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
            time_cost = time.time() - start_time
            time_cost_str = "{}h {}m {}s".format(int(time_cost // 3600), int(time_cost % 3600 // 60), int(time_cost % 60))
            tqdm.write('Epoch {} - Train Loss: {:.4f} - Val Loss: {:.4f}; time: {}; Epoch time cost:{}'.format(self.cur_epoch, train_loss, val_loss, 
                                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time_cost_str))

        # 保存TensorBoard日志文件
        if _TEST_TRAINER_:
            self.logger.writer.flush()
            self.logger.writer.close()
        if self.distributed:
            dist.destroy_process_group()
        if self.config.get("empty_cache", False):
            torch.cuda.empty_cache()
    
    def val(self):
        print("start to val... time:{}".format(self.start_timestamp))
        self.cur_epoch = 1
        num_workers = self.config.get("num_workers", 0)
        persistent_workers = self.config.get("persistent_workers", True)
        persistent_workers = False if num_workers == 0 else persistent_workers
        val_dataloader   = self._dataloader_type(self.val_dataset,   batch_size=self.batch_size, shuffle=False, 
                                      collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)
        
        start_time = time.time()
        tqdm.write('\nValidation start... time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        val_loss = self.val_one_epoch(val_dataloader)
        print("val_loss: {}".format(val_loss))
        time_cost = time.time() - start_time
        time_cost_str = "{}h {}m {}s".format(int(time_cost // 3600), int(time_cost % 3600 // 60), int(time_cost % 60))
        tqdm.write('Val Loss: {:.4f}; time: {}; Epoch time cost:{}'.format(val_loss, 
                                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time_cost_str))
        
