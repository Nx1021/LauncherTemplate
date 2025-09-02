from typing import Generator, Callable, Iterable, Union

from .BaseLauncher import Launcher, BaseLogger, collate_fn_decorator, _DatasetWrapper
# from .BaseLauncher import Launcher, BaseLogger, _DatasetWrapper

from . import load_yaml

import torch
from torch import Tensor
from torch.nn import Module
import torch.multiprocessing as mp

import abc
from typing import Optional, TypedDict, Generic, TypeVar, Any
import time
import datetime
from tqdm import tqdm
import os

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
DATASET_TYPE = TypeVar("DATASET_TYPE", bound=torch.utils.data.Dataset)
class Predictor(Launcher[MODEL_TYPE], Generic[MODEL_TYPE, DATASET_TYPE]):
    def __init__(self, 
                 model:MODEL_TYPE, 
                 config, 
                 dataset:Optional[DATASET_TYPE] = None,
                 log_dir = None):
        super().__init__(model, config, log_dir)
        self.config:PredictorConfig = config
        self._dataset = _DatasetWrapper(dataset)

        self.logger = BaseLogger(self.log_dir)
        self.logger.log(config)
        
        self.metric_function:Callable[[Any, Any], dict[str, tuple[int, int]]] = None
        self.precision_recorder = PrecisionRecorder()

        self.if_save_result     = self.config.get("if_save_result", False)
        self.num_saving_workers = self.config.get("num_saving_workers", 0)
        self.skip_exception     = self.config.get("skip_exception", False)
    
    @property
    def save_result_dir(self):
        return os.path.join(self.log_dir, "results")

    @property
    def dataset(self) -> DATASET_TYPE:
        return self._dataset.dataset
    
    @dataset.setter
    def dataset(self, value:DATASET_TYPE):
        assert isinstance(value, torch.utils.data.Dataset), "train_dataset should be a torch.utils.data.Dataset object"
        self._dataset = _DatasetWrapper(value)
    
    @Launcher.timing(-1)
    def example(self):
        pass

    @abc.abstractmethod
    def run_model(self, datas:list):
        pass

    def save_results(self, idx, results):
        pass
    
    def _save_results_processor(self, queue, worker_id):
        while True:
            data = queue.get()  # 获取数据
            if data is None:
                break  # 终止进程

            batch_idx, batch_data = data
            self.save_results(batch_idx, batch_data)

    def predict_all(self):
        if self.dataset is None:
            print("No dataset provided, cannot predict")
            return

        num_workers         = self.num_workers
        persistent_workers  = False
        dataloader   = self._dataloader_type(self._dataset,   batch_size=self.batch_size, shuffle=False, 
                                    collate_fn=self.collate_fn, num_workers = num_workers, pin_memory=False, persistent_workers = persistent_workers)
        dataloader.collate_fn = collate_fn_decorator(dataloader.collate_fn)
        
        self.model.eval()
        self.precision_recorder.clear() # clear the recorder

        progress = tqdm(dataloader, desc="Predict", leave=True)

        if self.if_save_result and self.num_saving_workers > 0:
            # 启动多个进程保存预测结果
            mp.set_start_method('spawn', force=True)  # 适用于 Windows & Linux
            queue = mp.Queue()
            workers = []
            for worker_id in range(self.num_saving_workers):
                p = mp.Process(target=self._save_results_processor, args=(queue, worker_id))
                p.start()
                workers.append(p)

        tqdm.write('\nPrediction start... time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        start_time = time.time()
        torch.set_grad_enabled(False)
        for datas, idx in progress:
            try:
                results = self.run_model(datas)
                if self.metric_function is not None:
                    self.precision_recorder.record(**self.metric_function(datas, results))
                progress.set_postfix({'Precision': self.precision_recorder.print()})

                # 预测结果放入队列，交给子进程保存
                if self.if_save_result:
                    if self.num_saving_workers > 0:
                        queue.put((idx, results))
                    else:
                        self.save_results(idx, results)
            except Exception as e:
                if self.skip_exception:
                    print("Error in prediction, skipping this batch")
                else:
                    raise e
                continue
        torch.set_grad_enabled(True)
        # 发送终止信号
        if self.if_save_result and self.num_saving_workers > 0:
            for _ in range(self.num_saving_workers):
                queue.put(None)

            # 等待所有 worker 进程完成
            for p in workers:
                p.join()
            
        time_cost = time.time() - start_time
        time_cost_str = "{}h {}m {}s".format(int(time_cost // 3600), int(time_cost % 3600 // 60), int(time_cost % 60))
        tqdm.write("'Precision': {}; time: {}; Epoch time cost:{}".format(self.precision_recorder.print(), 
                                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time_cost_str))
        
    def predict_one(self, datas:list):
        results = self.run_model(datas)
        return results

class ModelFreePredictor(Predictor[Module, DATASET_TYPE], Generic[DATASET_TYPE]):
    def __init__(self, dataset = None, batch_size = 1, num_loading_workers = 0, num_saving_workers = 0):
        super().__init__(None, {}, dataset)
        self.batch_size = batch_size
        self.num_workers = num_loading_workers
        self.num_saving_workers = num_saving_workers
        self.if_save_result = True



