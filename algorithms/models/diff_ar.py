import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from time import time as infer_time
from multiprocessing import Process
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection
from transformers import Wav2Vec2Model, HubertModel
from pytorch_lightning import LightningModule

from utils.demo_utils import animate
from algorithms.models.base import BaseTrainer
from algorithms.models.architectures import DiffARDecNetwork, VQAutoEncoder
from algorithms.metrics.losses import MaskedLosses

class DIFF_AR(BaseTrainer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # define model
        # self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg.MODEL.AUDIO_ENCODER.model_name_or_path)
        self.audio_encoder = HubertModel.from_pretrained(cfg.MODEL.AUDIO_ENCODER.model_name_or_path)
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
        if cfg.MODEL.finetune_dec:
            print("Finetune VAE Decoder...")
            for param in self.autoencoder.decoder.parameters():
                param.requires_grad = True

        self.diff_ar_network = DiffARDecNetwork(
            cfg=cfg,
            **(cfg.MODEL).get("DIFF_AR_DEC", dict())
        )
        
        # define loss
        self.losses = MetricCollection({
                split: MaskedLosses(cfg=cfg, split=split)
                for split in ["losses_train", "losses_test", "losses_val",]
            })

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
        audio = train_batch['audio'] # audio.shape = [batch_size, seq_len, audio_dim]
        vertice = train_batch['vertice'] # vertice.shape = [batch_size, vert_len, n_verts]
        template = train_batch['template'].unsqueeze(1) # template.shape = [batch_size, 1, n_verts]
        batch_size, vert_len, _ = vertice.shape
        hidden_state= self._audio2hidden(audio, length=vert_len)

        # window_size = torch.randint(2,self.cfg.MODEL.window_size+1,(1,))
        window_size = self.cfg.MODEL.window_size
        if window_size != 0:
            start_idx = torch.randint(0, vert_len-window_size-1,(1,)) if vert_len>window_size+1 else 0
            audio = audio[:,start_idx:start_idx+window_size]
            vertice = vertice[:,start_idx:start_idx+window_size]
            hidden_state = hidden_state[:,start_idx:start_idx+window_size]

        vertice_input = torch.cat((template,vertice), 1) # shift one position
        
        quant, emb_loss, info = self.autoencoder.encode(vertice_input-template)
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num, self.cfg.MODEL.VAE.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num * self.cfg.MODEL.VAE.zquant_dim).contiguous()

        diff_loss, x_pred = self.diff_ar_network(
            quant,
            hidden_state,
            train_batch['id']
        )

        x_pred = x_pred.permute(0,2,1)
        vertice_pred = self.autoencoder.decoder(x_pred) + template

        loss = self.losses['losses_train'].calculate_loss(vertice, vertice_pred, diff_loss)
        self.losses['losses_train'].update(loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        audio = val_batch['audio']
        vertice = val_batch['vertice']
        template = val_batch['template'].unsqueeze(1)
        hidden_state= self._audio2hidden(audio, length=vertice.shape[1])

        vertice_input = torch.cat((template,vertice), 1) # shift one position

        quant, emb_loss, info = self.autoencoder.encode(vertice_input-template)
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num, self.cfg.MODEL.VAE.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num * self.cfg.MODEL.VAE.zquant_dim).contiguous()
        
        batch_size = vertice.shape[0]
        id_dim = self.cfg.MODEL.DIFF_AR_DEC.id_dim
        # collect the results for each id
        loss_list = []

        # TODO: change to id_dim after ablation
        for idx in range(1):
            val_batch["id"] = torch.zeros(batch_size, id_dim).to(val_batch["vertice"].device)
            val_batch["id"][:, idx] = 1
            with torch.no_grad():
                # same as the training, we use the autoregressive inference
                diff_loss, x_pred = self.diff_ar_network.sample_fixed(
                    quant,
                    hidden_state,
                    val_batch['id'],
                    length=self.cfg.MODEL.window_size
                )

                x_pred = x_pred.permute(0,2,1)
                vertice_pred = self.autoencoder.decoder(x_pred) + template

                loss = self.losses['losses_val'].calculate_loss(vertice, vertice_pred, diff_loss)
                self.losses['losses_val'].update(loss)

                if loss is None:
                    return ValueError("loss is None")
                
                loss_list.append(loss)

            # visualize the result for the first id and the first batch
            if batch_idx == 0 and idx == 0:
                self._visualize(val_batch, vertice_pred)

        loss = torch.stack(loss_list, dim=0).mean(dim=0)        

        return loss
    
    def test_step(self, test_batch, batch_idx):
        audio = test_batch['audio']
        vertice = test_batch['vertice']
        template = test_batch['template'].unsqueeze(1)

        batch_size = vertice.shape[0]
        id_dim = self.cfg.MODEL.DIFF_AR_DEC.id_dim
                
        vocaset_lip_upper_ids = np.array((2865, 2866, 3546, 1751, 1750), dtype=np.int64)
        vocaset_lip_downer_ids = np.array((2907, 2938, 3504, 1849, 1804), dtype=np.int64)
        
        biwi_lip_upper_ids = np.array((366, 12773, 21688, 10235, 10117), dtype=np.int64)
        biwi_lip_downer_ids = np.array((21538, 10665, 22962, 21152, 14028), dtype=np.int64)

        vertice_input = torch.cat((template,vertice), 1) # shift one position

        quant, emb_loss, info = self.autoencoder.encode(vertice_input-template)
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num, self.cfg.MODEL.VAE.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num * self.cfg.MODEL.VAE.zquant_dim).contiguous()

        from utils.losses_utils import biwi_upper_face_variance, biwi_mouth_distance
        from utils.losses_utils import vocaset_upper_face_variance, vocaset_mouth_distance

        # collect the results for each id
        ndim = test_batch['id'].ndim
        if ndim == 2:
            test_batch_id_list = [test_batch['id']] # the id is given
        elif ndim == 3:
            test_batch_id_list = []
            for i in range(id_dim): # the id is not given, we use the id of each person in the training set
                test_batch["id"] = torch.zeros(batch_size, id_dim).to(test_batch["vertice"].device)
                test_batch["id"][:, i] = 1
                test_batch_id_list.append(test_batch['id'])
        else:
            raise ValueError(f"the dimension of the id should be 2 or 3, but got {ndim}")

        # collect the results for each id
        metrics_list = {}
        for idx, test_batch_id in enumerate(test_batch_id_list):

            test_batch["id"] = test_batch_id.to(test_batch["vertice"].device)

            with torch.no_grad():
                hidden_state= self._audio2hidden(audio, length=vertice.shape[1])

                # same as the validation, we use the autoregressive inference
                diff_loss, x_pred = self.diff_ar_network.sample_fixed(
                        quant,
                        hidden_state,
                        test_batch['id']
                    )

                x_pred = x_pred.permute(0,2,1)

                vertice_pred = self.autoencoder.decoder(x_pred) + template

                # calculate the metrics
                if vertice_pred.shape[-1] == 70110: # BIWI
                    exp = 'BIWI'
                    pred = vertice_pred.view(-1, 23370, 3).detach().cpu().numpy()
                    gt = vertice.view(-1, 23370, 3).detach().cpu().numpy()
                    template_npy =  test_batch['template'].view(-1, 23370, 3).detach().cpu().numpy()
                    min_len = min(pred.shape[0], gt.shape[0])
                    pred = pred[:min_len, :]
                    gt = gt[:min_len, :]
           
                    # (N, V)
                    pred_lip_dis = np.linalg.norm(pred[:, biwi_lip_upper_ids, :] - pred[:, biwi_lip_downer_ids, :], axis=-1)
                    gt_lip_dis = np.linalg.norm(gt[:, biwi_lip_upper_ids, :] - gt[:, biwi_lip_downer_ids, :], axis=-1)
                    
                    metrics = {
                        "FDD": biwi_upper_face_variance( motion = (gt - template_npy), ) \
                            - biwi_upper_face_variance(motion = (pred - template_npy), ),
                        "Lip Vertex Error": biwi_mouth_distance(
                            vertices_gt=gt,
                            vertices_pred=pred,
                        ).mean(),
                        "MOD": torch.tensor(np.mean(np.abs(pred_lip_dis - gt_lip_dis), axis=1).mean()).float(),
                        "Length": torch.tensor(min_len).float(),
                    }
                else: # VOCASET
                    exp = 'vocaset'
                    pred = vertice_pred.view(-1, 5023, 3).detach().cpu().numpy()
                    gt = vertice.view(-1, 5023, 3).detach().cpu().numpy()
                    template_npy =  test_batch['template'].view(-1, 5023, 3).detach().cpu().numpy()
                    min_len = min(pred.shape[0], gt.shape[0])
                    pred = pred[:min_len, :]
                    gt = gt[:min_len, :]

                    # (N, V)
                    pred_lip_dis = np.linalg.norm(pred[:, vocaset_lip_upper_ids, :] - pred[:, vocaset_lip_downer_ids, :], axis=-1)
                    gt_lip_dis = np.linalg.norm(gt[:, vocaset_lip_upper_ids, :] - gt[:, vocaset_lip_downer_ids, :], axis=-1)

                    metrics = {
                        "FDD": vocaset_upper_face_variance( motion = (gt - template_npy), ) \
                            - vocaset_upper_face_variance(motion = (pred - template_npy), ),
                        "Lip Vertex Error": vocaset_mouth_distance(
                            vertices_gt=gt,
                            vertices_pred=pred,
                        ).mean(),
                        "MOD": torch.tensor(np.mean(np.abs(pred_lip_dis - gt_lip_dis), axis=1).mean()).float(),
                        "Length": torch.tensor(min_len).float(),
                    }

                # save the metrics
                if metrics is None:
                    return ValueError("metrics is None")
                
                for key in metrics: # collect the metrics for each id
                    if key not in metrics_list:
                        metrics_list[key] = []
                    metrics_list[key].append(metrics[key])

            # visualize the result for the first id and the first batch
            if batch_idx == 0 and idx == 0:
                self._visualize(test_batch, vertice_pred)

        # average the metrics for each id
        for key in metrics_list:
            metrics_list[key] = torch.stack(metrics_list[key], dim=0).mean(dim=0)
            
        return metrics_list
    
    def predict(self, input_data):
        audio = input_data['audio']
        id = input_data['id']
        template = input_data['template'][None, ...].unsqueeze(1)
        vertice =  input_data['vertice']

        hidden_state= self._audio2hidden(audio, length=None)

        if vertice is not None:
            vertice_input = torch.cat((template,vertice), 1)-template # shift one position
        else:
            vertice_input = torch.cat((template,template), 1)-template

        quant, emb_loss, info = self.autoencoder.encode(vertice_input)
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num, self.cfg.MODEL.VAE.zquant_dim).contiguous()
        quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num * self.cfg.MODEL.VAE.zquant_dim).contiguous()

        with torch.no_grad():
            # same as the training, we use the autoregressive inference
            diff_loss, x_pred = self.diff_ar_network.sample_fixed(
                quant,
                hidden_state,
                id
            )

            x_pred = x_pred.permute(0,2,1)
            vertice_pred = self.autoencoder.decoder(x_pred) + template

        return vertice_pred
    
    def streaming_inference(self, input_data, vertices_list):
        audio = input_data['audio']
        id = input_data['id']
        template = input_data['template'][None, ...].unsqueeze(1)

        with torch.no_grad():
            hidden_state= self._audio2hidden(audio, length=None)

            vertice_input = torch.cat((template,template), 1)-template
            quant, emb_loss, info = self.autoencoder.encode(vertice_input)
            quant = quant.permute(0,2,1)
            quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num, self.cfg.MODEL.VAE.zquant_dim).contiguous()
            quant = quant.view(quant.shape[0], -1, self.cfg.MODEL.VAE.face_quan_num * self.cfg.MODEL.VAE.zquant_dim).contiguous()

            batch_size, seq_len, latent_dim = hidden_state.shape
            device = hidden_state.device

            # style embedding and hidden state projection
            obj_embedding = self.diff_ar_network.obj_vector(torch.argmax(id, dim=1)).unsqueeze(1) # obj_embedding.shape = [batch, 1, latent_dim]
            hidden_state = self.diff_ar_network.audio_feature_map(hidden_state) # hidden_state.shape = [batch, seq_len=vert_len, latent_dim]
            
            # start token
            x_start = quant[:,0].unsqueeze(1)

            # set timesteps
            self.diff_ar_network.scheduler.set_timesteps(self.cfg.MODEL.num_inference_timesteps)
            timesteps = self.diff_ar_network.scheduler.timesteps.to(device, non_blocking=True)

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            extra_step_kwargs = {}
            extra_step_kwargs["eta"] = self.cfg.MODEL.eta

            for i in range(seq_len):
                start_time = infer_time()
                if i == 0: 
                    past_latents = x_start

                conditions = self.diff_ar_network.condition_forward(past_latents, hidden_state, obj_embedding)
                past_latents_len = past_latents.shape[1]
                # sample a white noise
                future_latents = torch.randn(
                    size = (batch_size, past_latents_len, self.diff_ar_network.vq_dim),
                    dtype = torch.float,
                    device = device
                )

                # scale init noise by sigma
                future_latents = future_latents * self.diff_ar_network.scheduler.init_noise_sigma

                # denoising steps
                for j, t in enumerate(timesteps):
                    future_xs = self.diff_ar_network.vq_latent_map(future_latents)

                    t_emb = self.diff_ar_network.time_emb(t[None,...]).unsqueeze(1)
                    curr_conditions = conditions + t_emb
                    
                    diffusion_output = self.diff_ar_network.denoiser(torch.cat((future_xs, curr_conditions), dim=-1))

                    latents_output = self.diff_ar_network.latent_vq_map(diffusion_output)

                    future_latents = self.diff_ar_network.scheduler.step(latents_output, t, future_latents, **extra_step_kwargs).prev_sample

                new_output = future_latents[:,-1,:].unsqueeze(1)
                past_latents = torch.cat((past_latents, new_output), 1)

                x_pred = past_latents.permute(0,2,1)
                vertice_pred = self.autoencoder.decoder(x_pred) + template
                vertice_pred = vertice_pred.squeeze().cpu().numpy()
                vertice_pred = vertice_pred[-1].reshape(vertice_pred.shape[1]//3, 3)

                end_time = infer_time()
                print(f"frame{i}: {end_time-start_time:.3f}s")

                vertices_list.put(vertice_pred)
    
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
    
    def _visualize(self, batch, vertices_pred, parrallel = True):
        """
        Visualize the result
        Args:
            val_batch (dict)
            vertices_pred: Tensor [batch_size, vert_len, vert_dim]
            parrallel (bool): if True, the visualization will be performed in the current thread,
                otherwise, the visualization will be performed in a new thread
        """
        # visualize the result only for the first data in the batch
        data_idx = 0

        audio_path = batch['file_path'][data_idx]
        vis_path = os.path.join(
            self.cfg.FOLDER_EXP,
            "visualization",
            "{}_{}.{}".format(
                time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), 
                audio_path.split("/")[-1].split(".")[0],
                'mp4'
            )
        )

        # visualize the result
        if not parrallel:
            # use the current thread to visualize the result
            animate(
                vertices = vertices_pred[data_idx, ...].squeeze().cpu().numpy(),
                wav_path = audio_path,
                file_name = vis_path,
                ply = self.cfg.DEMO.PLY,
                fps = self.cfg.DEMO.FPS,
            )
        else:
            # use another thread to visualize the result
            p = Process(
                target=animate,
                args=(
                    vertices_pred[data_idx, ...].squeeze().cpu().numpy(),
                    audio_path,
                    vis_path,
                    self.cfg.DEMO.PLY,
                    self.cfg.DEMO.FPS,
                    ),
                )
            p.start()