import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW, Adam

from algorithms.metrics.losses import VAELosses
from algorithms.models.base import BaseTrainer
from algorithms.models.architectures import VQAutoEncoder

class VAETrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

        # define loss
        self.losses = MetricCollection({
                split: VAELosses(cfg=cfg, split=split)
                for split in ["losses_train", "losses_test", "losses_val",]
            })
        
        # define model
        self.autoencoder = VQAutoEncoder(cfg=cfg)

        # define optimizer
        if cfg.TRAIN.OPTIMIZER.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIMIZER.LR,
                                   params=filter(lambda p: p.requires_grad,self.parameters())
                                   )
        elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == "adam":
            self.optimizer = Adam(lr=cfg.TRAIN.OPTIMIZER.LR,
                                  params=filter(lambda p: p.requires_grad,self.parameters())
                                  )
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")
        
        # define scheduler
        if cfg.TRAIN.SCHEDULER.use:
            self.scheduler = StepLR(self.optimizer, step_size=cfg.TRAIN.SCHEDULER.step_size, gamma=cfg.TRAIN.SCHEDULER.gamma)
        else:
            self.scheduler = None

    def training_step(self, train_batch, batch_idx):
        vertice = train_batch['vertice']
        template = train_batch['template']

        vertice_pred, quant_loss, info = self.autoencoder(vertice, template)

        loss = self.losses['losses_train'].calculate_loss(vertice_pred, vertice, quant_loss)
        self.losses['losses_train'].update(loss)

        return loss

    
    def validation_step(self, val_batch, batch_idx):
        vertice = val_batch['vertice']
        template = val_batch['template']

        with torch.no_grad():
            vertice_pred, quant_loss, info = self.autoencoder(vertice, template)

            loss = self.losses['losses_val'].calculate_loss(vertice_pred, vertice, quant_loss)
            self.losses['losses_val'].update(loss)

        return loss