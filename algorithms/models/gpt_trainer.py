import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from multiprocessing import Process
from torch.distributions import Categorical
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR

from torchmetrics import MetricCollection
from transformers import Wav2Vec2Model

from utils.demo_utils import animate
from algorithms.models.base import BaseTrainer
from algorithms.models.architectures import VQAutoEncoder, Audio2MotionTransformer
from algorithms.metrics.losses import MaskedLosses,GPTLosses

class GptTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.mot_end_idx = cfg.MODEL.VAE.n_embed

        # define loss
        self.losses = MetricCollection({
                split:GPTLosses(cfg=cfg, split=split)
                for split in ["losses_train", "losses_test", "losses_val",]
            })
        self.loss_ce = torch.nn.CrossEntropyLoss()
        
        # define model
        self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg.MODEL.AUDIO_ENCODER.model_name_or_path)
        if cfg.MODEL.AUDIO_ENCODER.freeze:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        else:
            self.audio_encoder.feature_extractor._freeze_parameters()

        self.autoencoder = VQAutoEncoder(cfg=cfg)
        checkpoint = torch.load(cfg.MODEL.VAE.pretrained, map_location=self.audio_encoder.device)['state_dict']
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace('autoencoder.', '')
            new_state_dict[new_key] = value
        self.autoencoder.load_state_dict(new_state_dict)
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.a2m_gpt = Audio2MotionTransformer(cfg, **(cfg.MODEL).get("A2M_TRANS", dict()))

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
        self.scheduler = StepLR(self.optimizer, step_size=cfg.TRAIN.SCHEDULER.step_size, gamma=cfg.TRAIN.SCHEDULER.gamma)

    def training_step(self, train_batch, batch_idx):
        id = train_batch['id']
        audio = train_batch['audio']
        vertice = train_batch['vertice']
        template = train_batch['template'].unsqueeze(1)
        device = vertice.device

        face_quan_num = self.cfg.MODEL.VAE.face_quan_num
        batch_size, vert_len, _ = vertice.shape
        hidden_state= self._audio2hidden(audio, length=vert_len)

        quant, emb_loss, info = self.autoencoder.encode(vertice-template)
        perplexity, min_encodings, gt_indices = info

        target = gt_indices.view(batch_size, -1) # (batch, seq_len*h_face)
        # target = torch.concatenate([gt_indices, torch.ones((batch_size, face_quan_num), dtype=int,device=device) * self.mot_end_idx], axis=1)

        input_index = target[:,:-face_quan_num]

        mask = torch.bernoulli(self.cfg.MODEL.A2M_TRANS.prob * torch.ones(input_index.shape,device=device))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randint_like(input_index, self.cfg.MODEL.VAE.n_embed)
        a_indices = mask*input_index+(1-mask)*r_indices

        cls_pred = self.a2m_gpt(a_indices, hidden_state, id)
        cls_pred = cls_pred.contiguous()

        right_num = 0
        loss_cls = 0.0
        for i in range(batch_size):
            # loss function     (26), (26, 513)
            # print(cls_pred.shape)
            loss_cls += self.loss_ce(cls_pred[i][:(vert_len+1)*face_quan_num], target[i][:(vert_len+1)*face_quan_num]) / batch_size
            # Accuracy
            probs = torch.softmax(cls_pred[i][:(vert_len+1)*face_quan_num], dim=-1)
            # print(probs.shape)
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
            right_num += (cls_pred_index.flatten(0) == target[i][:(vert_len+1)*face_quan_num].flatten(0)).sum().item()

            index_pred = cls_pred_index

        # print(index_pred)
        quant_z = self.autoencoder.entry_to_feature(index_pred, (batch_size, self.cfg.MODEL.VAE.zquant_dim, -1))
        vertice_pred = self.autoencoder.decode(quant_z) + template
        acc = right_num * 100 / vert_len*face_quan_num
        
        loss = self.losses['losses_train'].calculate_loss(vertice, vertice_pred, acc, loss_cls)
        self.losses['losses_train'].update(loss)

        return loss

    
    def validation_step(self, val_batch, batch_idx):
        audio = val_batch['audio']
        vertice = val_batch['vertice']
        template = val_batch['template'].unsqueeze(1)
        device = vertice.device

        batch_size, vert_len, _ = vertice.shape
        hidden_state= self._audio2hidden(audio, length=vert_len)

        id = torch.zeros(batch_size, 8).to(val_batch["vertice"].device)
        id[:, 0] = 1

        right_num = 0
        loss_cls = torch.tensor(0.0)
        with torch.no_grad():
            vertice_pred, quant_loss, info = self.autoencoder.encode(vertice-template)
            perplexity, min_encodings, gt_indices = info
            gt_indices = gt_indices.view(batch_size, -1) # (batch, seq_len*h_face)

            cls_pred = self.a2m_gpt.sample(hidden_state, id)
            cls_pred = cls_pred.contiguous()
            # loss_cls += self.loss_ce(cls_pred[0][:(vert_len+1)], gt_indices[0][:(vert_len+1)]) / batch_size

            right_num += (cls_pred.flatten(0) == gt_indices[0][:(vert_len+1)].flatten(0)).sum().item()
            quant_z = self.autoencoder.entry_to_feature(cls_pred, (batch_size, self.cfg.MODEL.VAE.zquant_dim, -1))
            vertice_pred = self.autoencoder.decode(quant_z) + template
            acc = right_num * 100 / vert_len

            loss = self.losses['losses_val'].calculate_loss(vertice, vertice_pred, acc, loss_cls)
            self.losses['losses_val'].update(loss)

        return loss
    
    def _audio2hidden(self, audio, length=None):
        """
        Transfer audio to hidden state and interpolate(resize) to seq_len
        Args:
            audio: Tensor
            length: if None, length is calculated by input&output fps
        """
        input_fps = self.cfg.MODEL.AUDIO_ENCODER.audio_fps
        output_fps = self.cfg.MODEL.AUDIO_ENCODER.hidden_fps

        hidden_state = self.audio_encoder(audio).last_hidden_state
        seq_len = hidden_state.shape[1]

        hidden_state = hidden_state.transpose(1,2)
        if length is None:
            length = int(seq_len * (output_fps / input_fps))
        hidden_state = F.interpolate(hidden_state, 
                                     size = length, 
                                     align_corners=True, 
                                     mode="linear")
        
        return hidden_state.transpose(2,1)
    