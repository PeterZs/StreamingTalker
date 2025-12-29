import os
import time
import torch
from typing import List
import numpy as np

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

class BaseTrainer(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define loss

        # define optimizer

        # define model

    def training_epoch_end(self, outputs):
        callback_metrics = {}

        losses = self.losses["losses_train"]
        loss_dict = losses.compute("losses_train")
        losses.reset()
        callback_metrics.update({
            losses.loss2logname(loss, 'train'): value.item()
            for loss, value in loss_dict.items() #if not torch.isnan(value)
        })

        callback_metrics.update({
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch),
        })

        if not self.trainer.sanity_checking:
            self.log_dict(callback_metrics, sync_dist=True, rank_zero_only=True)

    def validation_epoch_end(self, outputs):
        callback_metrics = {}

        losses = self.losses["losses_val"]
        loss_dict = losses.compute("losses_val")
        losses.reset()
        callback_metrics.update({
            losses.loss2logname(loss, 'val'): value.item()
            for loss, value in loss_dict.items() #if not torch.isnan(value)
        })

        callback_metrics.update({
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch),
        })

        if not self.trainer.sanity_checking:
            self.log_dict(callback_metrics, sync_dist=True, rank_zero_only=True)

    def test_epoch_end(self, outputs):
        callback_metrics = {}

        metircs = {key: [] for key in outputs[0].keys()}
        for output in outputs: # collect the results from all batches
            for key, value in output.items():
                metircs[key].append(value)

        lengths = torch.stack(metircs.pop("Length"))
        for key, value in metircs.items():
            if key == 'Lip Vertex Error':
                metircs[key] = torch.mean(torch.stack(value) * lengths) / torch.mean(lengths)
            metircs[key] = torch.mean(torch.stack(value))
        callback_metrics.update(metircs)

        if not self.trainer.sanity_checking:
            self.log_dict(callback_metrics, sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                },
            },
        else:
            return self.optimizer