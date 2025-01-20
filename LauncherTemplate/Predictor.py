from typing import Generator, Callable, Iterable, Union

from .BaseLauncher import Launcher, BaseLogger
from . import load_yaml

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ConstantLR
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import abc
from typing import Optional, TypedDict, Generic, TypeVar, Any
import time
import datetime
from tqdm import tqdm

class PredictorConfig(TypedDict):
    device:str      = "cuda"

    batch_size:int  = 8
    num_workers:int = 4
    persistent_workers:bool = False

    distributed:bool = False

class PrecisionRecorder():
    def __init__(self) -> None:
        self.total_num:dict[str] = {}
        self.TP_num:dict[str] = {}

    def record(self, **kwargs:tuple[int, int]):
        for name, numbers in kwargs.items():
            self.total_num[name]    = self.total_num.get(name, 0)   + numbers[0]
            self.TP_num[name]       = self.TP_num.get(name, 0)      + numbers[1]
    
    def print(self):
        strings = []
        for name in self.total_num:
            Precison = self.TP_num[name] / self.total_num[name]
            strings.append("{}: {:.4f}".format(name, Precison))
        if len(strings) == 0:
            return "No data recorded"
        else:
            return ", ".join(strings)
    
    def clear(self):
        self.total_num.clear()
        self.TP_num.clear()

MODEL_TYPE = TypeVar("MODEL_TYPE", bound=torch.nn.Module)
class Predictor(Launcher[MODEL_TYPE], Generic[MODEL_TYPE]):
    def __init__(self, 
                 model:MODEL_TYPE, 
                 config, 
                 dataset:Optional[torch.utils.data.Dataset] = None,
                  log_remark = ""):
        super().__init__(model, config, log_remark)
        self.config:PredictorConfig = config
        self.dataset:torch.utils.data.Dataset = dataset

        self.logger = BaseLogger(self.log_dir)
        self.logger.log(config)
        
        self.metric_function:Callable[[Any], dict[str, tuple[int, int]]] = None
        self.precision_recorder = PrecisionRecorder()

    @Launcher.timing(-1)
    def example(self):
        pass

    @abc.abstractmethod
    def run_model(self, datas:list) -> Union[None, Tensor]:
        pass
    
    def predict_all(self):
        if self.dataset is None:
            print("No dataset provided, cannot predict")
            return

        num_workers         = self.config.get("num_workers", 0) # default 0
        persistent_workers  = False
        dataloader   = self._dataloader_type(self.dataset,   batch_size=self.batch_size, shuffle=False, 
                                    collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=True, persistent_workers = persistent_workers)
        self.precision_recorder.clear() # clear the recorder

        progress = tqdm(dataloader, desc="Predict", leave=True)
        tqdm.write('\nPrediction start... time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        start_time = time.time()
        for datas in progress:
            results = self.run_model(datas)
            if self.metric_function is not None:
                self.precision_recorder.record(self.metric_function(results))
            progress.set_postfix({'Precision': self.precision_recorder.print()})
            
        time_cost = time.time() - start_time
        time_cost_str = "{}h {}m {}s".format(int(time_cost // 3600), int(time_cost % 3600 // 60), int(time_cost % 60))
        tqdm.write("'Precision': {}; time: {}; Epoch time cost:{}".format(self.precision_recorder.print(), 
                                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time_cost_str))
        



