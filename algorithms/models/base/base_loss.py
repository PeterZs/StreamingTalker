import os
import pickle
import torch
import numpy as np
import torch.nn as nn

from torchmetrics import Metric

class BaseLosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set up loss 
        self.losses = []
        self._losses_func = {}
        self._params = {}

        name = "total"
        self.losses.append(name)
        self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        name = 'count'
        self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")
        
    
    # Call this function in forward process, return total loss
    def calculate_loss(self, vertice_gt, vertice_pred, diff_loss=None, mask=None) -> torch.Tensor:
        total = torch.tensor(0.0, dtype=torch.float32, device=vertice_gt.device)

        # define your own loss here

        return total
        
    # NO Returns
    def update(self, ttl) -> None:
        if self.split in ['losses_train', 'losses_val']: 
            self.total += ttl.detach()
            self.count += 1

        elif self.split in ['losses_test']:
            raise ValueError(f"Test split don't need loss")

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}


    def _update_loss(self, loss: str, outputs, inputs, mask = None):
        # Update the loss
        if mask is not None:
            val = self._losses_func[loss](outputs, inputs, mask)
        else:
            val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        
        # define your log name
        log_name = None

        return log_name