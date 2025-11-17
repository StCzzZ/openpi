import torch
from typing import Callable, Dict, Any
import math
from torch.optim import Optimizer

def dict_apply(dict_data: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    for key, value in dict_data.items():
        if isinstance(value, dict):
            dict_data[key] = dict_apply(value, func)
        else:
            dict_data[key] = func(value)
    return dict_data



class CosineWithMinLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            # Cosine decay for the first max_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos_decay for _ in self.base_lrs]
        else:
            # Keep the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]