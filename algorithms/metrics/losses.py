import os
import pickle
import torch
import numpy as np
import torch.nn as nn

from algorithms.models.base import BaseLosses

class GPTLosses(BaseLosses):
    '''
    GPT Loss Function for stage2
    Params:
        cfg: OmegaConf config file
        split: train, val or test
    '''
    def __init__(self, cfg, split):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        self.cfg = cfg

        reconstruct = MaskedConsistency()
        reconstruct_v = MaskedVelocityConsistency()

        self.split = split
        is_train = split in ['losses_train']

        if split in ['losses_train', 'losses_val']:
            name = "cross_entropy"
            self.losses.append(name)
            self._losses_func[name] = nn.L1Loss(reduction="mean")
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "acc"
            self.losses.append(name)
            self._losses_func[name] = lambda x: x
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            # vertice 
            name = "vertice_enc" # enc here means encoding, just for matching the name in the the diffusion-denoising experiment
            self.losses.append(name)
            self._losses_func[name] = reconstruct
            self._params[name] =  1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "vertice_encv" # encv here means encoding velocity, just for matching the name in the the diffusion-denoising experiment
            self.losses.append(name)
            self._losses_func[name] = reconstruct_v
            self._params[name] =  1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        elif split in ['losses_test']:
            pass # no loss for test
        else:
            raise ValueError(f"split {split} not supported") 
    
    # Call this function in forward process, return total loss
    def calculate_loss(self, vertice_gt, vertice_pred, acc, cross_entropy , mask=None) -> torch.Tensor:
        total = torch.tensor(0.0, dtype=torch.float32, device=vertice_gt.device)

        self.cross_entropy += cross_entropy.detach()
        self.acc += acc

        # vertice loss
        total += self._update_loss("vertice_enc", vertice_gt, vertice_pred, mask = mask)
        total += self._update_loss("vertice_encv", vertice_gt, vertice_pred, mask = mask)

        total += cross_entropy

        return total

    def loss2logname(self, loss: str, split: str):
        log_name = f"{loss}/{split}"

        return log_name

class VAELosses(BaseLosses):
    '''
    VAE Loss Function for stage1: 
        Sum of reconstruction loss and embed loss,
        (the latter item has stop-gradient operator and is computed by VectorQuantizer)
    Params:
        cfg: OmegaConf config file
        split: train, val or test
    '''
    def __init__(self, cfg, split):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        self.cfg = cfg

        self.split = split
        is_train = split in ['losses_train']

        if split in ['losses_train', 'losses_val']:
            # reconstruction loss
            name = "recon_loss"
            self.losses.append(name)
            self._losses_func[name] = nn.L1Loss(reduction="mean")
            self._params[name] = cfg.LOSS.RECON_LOSS if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            # embed loss
            name = "embed_loss"
            self.losses.append(name)
            self._losses_func[name] = lambda x: x
            self._params[name] = cfg.LOSS.EMBED_LOSS if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        elif split in ['losses_test']:
            pass # no loss for test
        else:
            raise ValueError(f"split {split} not supported") 
    
    # Call this function in forward process, return total loss
    def calculate_loss(self, pred, target, quant_loss, mask=None) -> torch.Tensor:
        total = torch.tensor(0.0, dtype=torch.float32, device=pred.device)

        # embed loss
        total += quant_loss*self._params['embed_loss']
        self.embed_loss += quant_loss.detach()

        # reconstruction loss
        total += self._update_loss("recon_loss", pred, target, mask = mask)

        return total

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{split}"
        return log_name

# Modified from DiffSpeaker: https://github.com/theEricMa/DiffSpeaker
class MaskedLosses(BaseLosses):
    """
    MLD Loss
    """

    def __init__(self, cfg, split):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        self.cfg = cfg

        reconstruct = MaskedConsistency()
        reconstruct_v = MaskedVelocityConsistency()

        self.split = split
        is_train = split in ['losses_train']

        if split in ['losses_train', 'losses_val']:
            # diff loss
            name = "diff_loss" 
            self.losses.append(name)
            self._losses_func[name] = lambda x: x
            self._params[name] = cfg.LOSS.DIFF_LOSS
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            # vertice 
            name = "vertice_enc" # enc here means encoding, just for matching the name in the the diffusion-denoising experiment
            self.losses.append(name)
            self._losses_func[name] = reconstruct
            self._params[name] = cfg.LOSS.VERTICE_ENC if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "vertice_encv" # encv here means encoding velocity, just for matching the name in the the diffusion-denoising experiment
            self.losses.append(name)
            self._losses_func[name] = reconstruct_v
            self._params[name] = cfg.LOSS.VERTICE_ENC_V if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "lip_enc"
            self.losses.append(name)
            self._losses_func[name] = reconstruct
            self._params[name] = cfg.LOSS.LIP_ENC if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "lip_encv"
            self.losses.append(name)
            self._losses_func[name] = reconstruct_v
            self._params[name] = cfg.LOSS.LIP_ENC_V if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        elif split in ['losses_test']:
            pass # no loss for test
        else:
            raise ValueError(f"split {split} not supported")

        # obtain the lmk index
        if cfg.LOSS.TYPE == "biwi":
            # the following is required by BIWI
            with open(os.path.join(cfg.DATASET.DATA_ROOT, "regions", "lve.txt")) as f:
                maps = f.read().split(", ")
                self.biwi_mouth_map = [int(i) for i in maps]
                self.biwi_mouth_map = torch.tensor(self.biwi_mouth_map).long()
        elif cfg.LOSS.TYPE == "vocaset":
            # the following is required by vocaset
            with open(os.path.join(cfg.DATASET.DATA_ROOT, "FLAME_masks.pkl"), "rb") as f:
                masks = pickle.load(f, encoding='latin1')
                self.vocaset_mouth_map = masks["lips"].tolist()     
                self.vocaset_mouth_map = torch.tensor(self.vocaset_mouth_map).long()   

    def vert2lip(self, vertice):
        num_verts = vertice.shape[-1] // 3
        if num_verts == 5023:
            mouth_map = self.vocaset_mouth_map.to(vertice.device)
        elif num_verts == 23370:
            mouth_map = self.biwi_mouth_map.to(vertice.device)
        else:
            raise ValueError(f"num_verts {num_verts} not supported")
        
        shape = vertice.shape
        lip_vertice = vertice.view(shape[0], shape[1], -1, 3)[:, :, mouth_map, :].view(shape[0], shape[1], -1)
        return lip_vertice
    
    # Call this function in forward process, return total loss
    def calculate_loss(self, vertice_gt, vertice_pred, diff_loss=None, mask=None) -> torch.Tensor:
        total = torch.tensor(0.0, dtype=torch.float32, device=vertice_gt.device)

        # diffusion loss
        if diff_loss is not None:
            total += diff_loss * self._params['diff_loss']
            self.diff_loss += diff_loss.detach()

        # vertice loss
        total += self._update_loss("vertice_enc", vertice_gt, vertice_pred, mask = mask)
        total += self._update_loss("vertice_encv", vertice_gt, vertice_pred, mask = mask)
        
        # lip loss
        lip_vertice = self.vert2lip(vertice_gt)
        lip_vertice_pred = self.vert2lip(vertice_pred)
        total += self._update_loss("lip_enc", lip_vertice, lip_vertice_pred, mask = mask)
        total += self._update_loss("lip_encv", lip_vertice, lip_vertice_pred, mask = mask)

        return total

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        elif loss == "diff_loss":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
    
class MaskedConsistency:
    def __init__(self) -> None:
        self.loss = nn.MSELoss(reduction="mean")

    def __call__(self, gt, pred, mask=None):
        # # masking nan
        # is_nan = torch.logical_or(torch.isnan(pred), torch.isnan(gt))
        # nan_mask = torch.logical_not(is_nan).long()
        # torch.where(nan_mask[0, ..., 0] != mask[0].squeeze())
        if mask is not None:
            return self.loss(mask * pred, mask * gt)
        else:
            return self.loss(pred, gt)
        
    def __repr__(self):
        return self.loss.__repr__()
    
class MaskedVelocityConsistency:
    def __init__(self) -> None:
        self.loss = nn.MSELoss(reduction="mean")

    def __call__(self, gt, pred, mask=None):
        if mask is not None:
            term1 = self.velocity(mask * pred)
            term2 = self.velocity(mask * gt)
            return self.loss(term1, term2)
        else:
            term1 = self.velocity(pred)
            term2 = self.velocity(gt)
            return self.loss(term1, term2)

    def velocity(self, term):
        velocity = term[:, 1:] - term[:, :-1]
        return velocity

    def __repr__(self):
        return self.loss.__repr__()