from typing import Generator, Callable, Iterable, Union

from .BaseLauncher import Launcher, BaseLogger
from . import load_yaml

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
