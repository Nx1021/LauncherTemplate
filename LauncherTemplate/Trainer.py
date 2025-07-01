import torch.utils
import torch.utils.data
from . import SCRIPT_DIR, WEIGHTS_DIR, load_yaml, LOGS_DIR
from .BaseLauncher import BaseLogger, Launcher, collate_fn_decorator, create_folder_shortcut, MODEL_TYPE, \
    _DatasetWrapper

import os
import win32com.client
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ConstantLR, LinearLR, StepLR
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

from typing import Union, TypedDict, Callable, Optional, Iterable, Generic, TypeVar, Literal
sys_name = platform.system()
if sys_name == "Windows":
    TESTFLOW = False
else:
    TESTFLOW = False

FUNC_NAMES = Literal["warmup", "cosine", "constant", "step"]

class TrainFlowKW(TypedDict):
    lr_func: FUNC_NAMES
    lr: float
    cfg: dict

class TrainFlow():
    """defination of training flow: how to change learning rate.
    """
    def __init__(self, trainer:"Trainer", flowfile:Union[str,dict[int, TrainFlowKW]]) -> None:
        """
        Args:
            trainer (Trainer): Trainer object
            flowfile (Union[str,dict[int, TrainFlowKW]]): a yaml file path that contains the training flow dict or a dict object

        Raises:
            ValueError: the first stage must start at epoch 0!

        Example:
            An example of flowfile:
            ```
            0:
                lr_func: warmup
                lr:      0.00002
                cfg:     {"A": 1}
            5:
                lr_func: constant
                lr:      0.00002
                cfg:
            30:
                lr_func: cosine
                lr:      0.00002
                cfg:
            41:
            ```

            In this version, 3 different learning rate functions are supported: warmup, cosine, constant:

            * warmup: linearly increase the learning rate from 0 to the specified value `lr`.
            * cosine: cosine annealing learning rate, the learning rate will decrease from the specified value `lr` to 0.
            * constant: keep the learning rate as the specified value `lr`.
        """
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

    def _get_lr_func(self, lr_name:FUNC_NAMES, totol_step:int, initial_lr:float):
        # totol_step = totol_step * int(np.round(len(self.trainer.train_dataset) / self.trainer.batch_size))
        for param_group in self.trainer.optimizer.param_groups:
            param_group['initial_lr'] = initial_lr
            param_group['lr'] = initial_lr
        if lr_name == "warmup":
            return LinearLR(self.trainer.optimizer, start_factor=1e-10, end_factor=1.0, total_iters=totol_step)
                # LambdaLR(self.trainer.optimizer, 
                #                     lr_lambda=lambda step: min(step / totol_step, 1.0))
        if lr_name == "cosine":
            return CosineAnnealingLR(self.trainer.optimizer, totol_step)
        if lr_name == "constant":
            return ConstantLR(self.trainer.optimizer, 1.0, 1)
        if lr_name == "step":
            gamma = np.power(1e-4, totol_step)
            return StepLR(self.trainer.optimizer, step_size=1, gamma=gamma)
    
    def _enter_new_stage(self):
        stage_info = self.flow[self.stage_segment[self.cur_stage]]
        if stage_info is None:
            return
        totol_step = self.stage_segment[self.cur_stage + 1] - self.stage_segment[self.cur_stage]
        
        if stage_info.get("lr_func") is not None and stage_info.get("lr") is not None:
            self.scheduler = self._get_lr_func(stage_info["lr_func"], totol_step, stage_info["lr"])

        if stage_info.get("cfg") is not None and isinstance(stage_info["cfg"], dict):
            self.trainer.config.update(stage_info["cfg"])

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.epoch >= self.stage_segment[-1]:
            raise StopIteration      
        if self.epoch in self.stage_segment:
            self._enter_new_stage()    
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
        # SummaryWriter
        self.writer.add_scalar(tag, value, step)

        # write to log file
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
    
    def comment(self, comment):
        if not self._on:
            return
        log_file = os.path.join(self.log_dir, "comment.txt")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = "{}  \t{}\n".format(current_time, comment)
        with open(log_file, 'a', encoding="utf-8") as f:
            f.write(log_line)

LOSSMODULE = TypeVar("LOSSMODULE", bound=nn.Module)
class LossManager(Generic[LOSSMODULE]):
    def __init__(self, loss:Optional[Callable] = None, losspart_names:Optional[Iterable[str]] = None) -> None:
        self.loss_module:LOSSMODULE = loss
        self._mode:bool = True
        self.losspart_names = losspart_names 
        if self.losspart_names is not None:
            self._init_loss_mngr(self.losspart_names)
        else:
            self._init_loss_mngr([])
        
        self._watch_on_inited = False
        self._init_watch_on_loss_mngr([])

    def _init_loss_mngr(self, losspart_names:Iterable[str]):
        self.train_loss_mngr = _LossManager(losspart_names)
        self.val_loss_mngr   = _LossManager(losspart_names)

    def _init_watch_on_loss_mngr(self, watch_on_keys:Iterable[str]):
        self._watch_on_train_loss_mngr = _LossManager(watch_on_keys)
        self._watch_on_val_loss_mngr   = _LossManager(watch_on_keys)

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
        
    @property
    def _watch_on_loss_mngr(self):
        if self._mode:
            return self._watch_on_train_loss_mngr
        else:
            return self._watch_on_val_loss_mngr
    
    def _get_batch_size(self, *args, **kwargs) -> int:
        return len(args[0])

    def __call__(self, *args, BATCHSIZE = None, **kwargs):
        B = self._get_batch_size(*args, **kwargs) if BATCHSIZE is None else BATCHSIZE
        if self.loss_module is None:
            losses = args[0]
        else:
            losses:Union[tuple[Tensor], dict[str, Tensor], Tensor, tuple[tuple, dict]] = self.loss_module(*args, **kwargs)
            if isinstance(losses, tuple) and len(losses) == 2 and isinstance(losses[1], dict):
                watch_on:dict[str, Tensor] = losses[1]
                losses = losses[0]
            else:
                watch_on = None
        with torch.no_grad():
            if isinstance(losses, (tuple, dict)):
                # multiple loss items
                if self.losspart_names is None:
                    self.losspart_names = [f"loss_{i}" for i in range(len(losses))]
                    self._init_loss_mngr(self.losspart_names)
                assert len(losses) == len(self.losspart_names), f"expect {len(self.losspart_names)} loss items, but got {len(losses)} items"
                if isinstance(losses, tuple):
                    losses = {k: v for k, v in zip(self.losspart_names, losses)}
                self.loss_mngr.record(losses, B)
            elif isinstance(losses, Tensor):
                if self.losspart_names is None:
                    self.losspart_names = ["loss"]
                    self._init_loss_mngr(self.losspart_names)
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

        if watch_on is not None:
            assert isinstance(watch_on, dict), f"watch_on should be a dict, but got {type(watch_on)}"
            watch_on_keys = list(watch_on.keys())
            if not self._watch_on_inited:
                self._init_watch_on_loss_mngr(watch_on_keys)
                self._watch_on_inited = True
            self._watch_on_loss_mngr.record(watch_on, B)

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
        self._last_loss = float(self.loss.item())
        # self._last_loss = torch.sum(torch.stack(losses)).item()
        _temp = {}
        for key, value in losses.items():
            if key not in self.__loss_names:
                raise ValueError(f"loss name:{key} not in {self.__loss_names}")
            new_losss = (self.loss_record[key] * self._total_num + value * num) / (self._total_num + num)
            _temp[key] = new_losss
        if any([x.isnan().item() for x in _temp.values()]):
            return
        else:
            for key in _temp.keys():
                self.loss_record[key] = _temp[key]
        self._total_num += num

        loss_parts = [v for k, v in self.loss_record.items() if k is not LossKW.LOSS]
        self.loss_record[LossKW.LOSS] = torch.sum(torch.stack(loss_parts))

    def clear(self):
        for k in list(self.loss_record.keys()):
            del self.loss_record[k]
            self.loss_record[k] = torch.tensor(0.0)
        self._total_num = 0

class TrainerConfig(TypedDict):
    subset_sampling_training_ratio: Optional[float] = None

    device:str      = "cuda"

    batch_size:int  = 8
    num_workers:int = 4
    shuffle_training:bool = True
    persistent_workers:bool = True
    empty_cache:bool = True

    start_epoch:int = 1
    train_flow:dict[int, TrainFlowKW] = {}
    distributed:bool = False

    # saving
    saving_period:int = 10

_TEST_TRAINER_ = False



MODEL_TYPE = TypeVar("MODEL_TYPE", bound=nn.Module)
LOSS_TYPE  = TypeVar("LOSS_TYPE", bound=nn.Module)
DATASET_TYPE = TypeVar("DATASET_TYPE", bound=torch.utils.data.Dataset)
class Trainer(Launcher[MODEL_TYPE], abc.ABC, Generic[MODEL_TYPE, LOSS_TYPE, DATASET_TYPE]):
    def __init__(self,
                 model:MODEL_TYPE,
                 train_dataset:DATASET_TYPE,
                 val_dataset:DATASET_TYPE,
                 criterion: Union[Callable, nn.Module, LossManager],
                 config:TrainerConfig):
        """
        Args:
            model (MODEL_TYPE|nn.Module): the model to be trained
            train_dataset (DATASET_TYPE|torch.utils.data.Dataset): dataset for training
            val_dataset   (DATASET_TYPE|torch.utils.data.Dataset): dataset for validation
            criterion (Union[Callable, nn.Module, LossManager]): loss module, it will be converted to a `LossManager` if it is not a `LossManager`
            config (TrainerConfig): 
        """
        super().__init__(model, config)
        self._train_dataset = _DatasetWrapper(train_dataset)
        self._val_dataset   = _DatasetWrapper(val_dataset)

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
        config = self.check_cfg(config)
        self.start_epoch    = config["start_epoch"]
        self.saving_period  = config["saving_period"]
        self._training_forward = True
        self._without_model = False

        self.__last_backup_time = None

        with open(os.path.join(self.log_dir, "Trainer"), "w") as f:
            pass

    @property
    def dataset(self) -> DATASET_TYPE:
        return self._train_dataset.dataset

    @property
    def train_dataset(self) -> DATASET_TYPE:
        return self._train_dataset.dataset
    
    @train_dataset.setter
    def train_dataset(self, value:DATASET_TYPE):
        assert isinstance(value, torch.utils.data.Dataset), "train_dataset should be a torch.utils.data.Dataset object"
        self._train_dataset = _DatasetWrapper(value)
    
    @property
    def val_dataset(self) -> DATASET_TYPE:
        return self._val_dataset
    
    @val_dataset.setter
    def val_dataset(self, value:DATASET_TYPE):
        assert isinstance(value, torch.utils.data.Dataset), "val_dataset should be a torch.utils.data.Dataset object"
        self._val_dataset = _DatasetWrapper(value)

    def check_cfg(self, config:TrainerConfig):
        if "subset_sampling_training_ratio" in config:
            assert 0 < config["subset_sampling_training_ratio"] <= 1, "subset_sampling_training_ratio should be in (0, 1]"
            config["shuffle_training"] = True

        return config

    @property
    def inner_model(self) -> nn.Module:
        if isinstance(self.model, torch.nn.DataParallel):
            module:nn.Module= self.model.module # type: ignore
            return module
        else:
            return self.model

    def _backup_weights(self):
        if self.__last_backup_time is None:
            self.__last_backup_time = time.time()
        if (time.time() - self.__last_backup_time) > 20 * 60:
            self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp + f"backup")
            self.__last_backup_time = time.time()

    def if_skip(self):
        return self.cur_epoch < self.start_epoch
    
    @abc.abstractmethod
    def run_model(self, datas:list) -> Union[None, Tensor]:
        """An abstract method for running model

        Args:
            datas (list): _description_

        Returns:
            Union[None, Tensor]: _description_
        """
        pass

    def epoch_preparation(self, epoch:int):
        """A hook for epoch preparation
        """
        pass
    
    def step_preparation(self, step:int):
        """A hook for step preparation
        """
        pass

    def backward(self, loss:Tensor):
        """a hook for backward propagation operation

        Args:
            loss (Tensor)
        """
        loss.backward()
        self.optimizer.step()

    def loss_nan_process(self, loss:Tensor):
        """a hook for loss nan process

        Args:
            loss (Tensor): 

        Returns:
            if_continue (bool): if continue to train. If True, continue to train, otherwise, stop training and exit.
        """
        print("loss nan, continue! zero grad")
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return True

    def save_model_checkpoints(self, save_dir, timestamp):
        """save model checkpoints
        
        Args:
            save_dir (str): the directory to save model checkpoints
            timestamp (str): 
        """
        if self._without_model:
            return 
        if self.distributed:
            torch.save(self.model.module.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_dir, timestamp + '_model.pth'))
        short_cut_path = os.path.join(WEIGHTS_DIR, self._log_sub_dir+'_training_logs.lnk')
        if not os.path.exists(short_cut_path):
            create_folder_shortcut(self.log_dir, short_cut_path, "link of logs")

    def forward_one_epoch(self, dataloader:DataLoader, _training_forward = False):
        """forward one epoch

        Args:
            dataloader (DataLoader): 
            _training_forward (bool, optional): Specify whether the forward is in training mode or validation mode. Defaults to False.

        Returns:
            _type_: _description_
        """
        self._training_forward = _training_forward
        desc = "Train" if _training_forward else "Val"

        LEN = len(dataloader)
        if _training_forward and self.config.get("subset_sampling_training_ratio") is not None:
            LEN = int(LEN * self.config["subset_sampling_training_ratio"])
        step_count = 0

        progress = tqdm(dataloader, desc=desc, total=LEN, leave=True)
        for datas, idx in progress:
            if step_count >= LEN:
                break
            step_count += 1

            self.step_preparation(step_count)

            if self._without_model:
                progress.set_postfix({"info:": "without model"})
                continue
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
            # backward propagation and optimization
            self.optimizer.zero_grad()
            if _training_forward and isinstance(loss, torch.Tensor) and loss.grad_fn is not None:
                self.backward(loss)

            # update progress bar
            _postfix = {'Loss': "{:>8.4f}".format(self.criterion.loss_mngr.last_loss_value)}
            for k, v in self.criterion._watch_on_loss_mngr.loss_record.items():
                if k == LossKW.LOSS:
                    continue
                _postfix[k] = "{:>4.4f}".format(v)
            progress.set_postfix(_postfix)
            if TESTFLOW:
                break
            
            if _training_forward:
                self._backup_weights()
            
        # 将val_loss写入TensorBoard日志文件
        if _training_forward:
            self.logger.write_epoch("Learning rate", self.optimizer.param_groups[0]["lr"], self.cur_epoch)
        for key, value in self.criterion.loss_mngr.loss_record.items():
            self.logger.write_epoch(self.criterion.mode_str + '-' + key, value, self.cur_epoch)
        for key, value in self.criterion._watch_on_loss_mngr.loss_record.items():
            if key == LossKW.LOSS:
                continue
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

    def train(self, without_model = False):
        print("start to train... time:{}".format(self.start_timestamp))

        self._without_model = without_model
        self.cur_epoch = 0
        # num_workers = 0 if sys_name == "Windows" else 4
        num_workers = self.num_workers
        persistent_workers = self.config.get("persistent_workers", True)
        persistent_workers = False if num_workers == 0 else persistent_workers
        shuffle_training = self.config.get('shuffle_training', True)
        train_dataloader = self._dataloader_type(self._train_dataset, batch_size=self.batch_size, shuffle=shuffle_training,  
                collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)
        val_dataloader   = self._dataloader_type(self._val_dataset,   batch_size=self.batch_size, shuffle=False, 
                collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)

        train_dataloader.collate_fn = collate_fn_decorator(train_dataloader.collate_fn)
        val_dataloader.collate_fn   = collate_fn_decorator(val_dataloader.collate_fn)

        for epoch in self.flow:
            self.cur_epoch = epoch
            start_time = time.time()
            tqdm.write("\nEpoch {} start... time: {}, Lr: {:>2.7f}".format(self.cur_epoch, 
                                                                           datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                           self.optimizer.param_groups[0]["lr"]))
            if not self.if_skip():
                self.epoch_preparation(epoch)
                
                # 训练阶段
                train_loss = self.train_one_epoch(train_dataloader)

                # 验证阶段
                val_loss = self.val_one_epoch(val_dataloader)

                # 如果验证损失低于历史最小值，则保存模型权重
                if val_loss < self.best_val_loss:
                    print("new best val_loss: {}, saving...".format(val_loss))
                    self.best_val_loss = val_loss
                    self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp)
                if epoch % self.saving_period == 0:
                    self.save_model_checkpoints(WEIGHTS_DIR, self.start_timestamp + f"_{self.cur_epoch}_")

                time_cost = time.time() - start_time
                time_cost_str = "{}h {}m {}s".format(int(time_cost // 3600), int(time_cost % 3600 // 60), int(time_cost % 60))
            else:
                train_loss = 0.0
                val_loss   = 0.0
                time_cost_str = "skipped"
            
            if self._without_model:
                break

            # 更新进度条信息
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
        self._without_model = False

        print("train finished")

    def val(self):
        print("start to val... time:{}".format(self.start_timestamp))
        self.cur_epoch = 1
        num_workers = self.config.get("num_workers", 0)
        persistent_workers = self.config.get("persistent_workers", True)
        persistent_workers = False if num_workers == 0 else persistent_workers
        val_dataloader   = self._dataloader_type(self.val_dataset,   batch_size=self.batch_size, shuffle=False, 
                                      collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)
        val_dataloader.collate_fn = collate_fn_decorator(val_dataloader.collate_fn)
        start_time = time.time()
        tqdm.write('\nValidation start... time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        with torch.no_grad():
            val_loss = self.val_one_epoch(val_dataloader)
        print("val_loss: {}".format(val_loss))
        time_cost = time.time() - start_time
        time_cost_str = "{}h {}m {}s".format(int(time_cost // 3600), int(time_cost % 3600 // 60), int(time_cost % 60))
        tqdm.write('Val Loss: {:.4f}; time: {}; Epoch time cost:{}'.format(val_loss, 
                                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time_cost_str))



# def collate_fn_decorator(collate_fn, launcher:Launcher):
#     def _collate_fn(batch):
#         datas   = [x[:-1] for x in batch]
#         indices = [x[-1] for x in batch]

#         datas = collate_fn(datas)
#         launcher._temp = indices
#         return datas
#     return _collate_fn