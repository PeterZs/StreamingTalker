import torch
import inspect
import torch.nn as nn
from time import time
from diffusers import DDIMScheduler, DDPMScheduler

from algorithms.models.base import BaseModel
from algorithms.models.architectures.common import PeriodicPositionalEncoding, TimestepEmbedder
from utils.model_utils import temporal_bias_mask, align_bias_mask, mean_flat, modulate

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class DiffARDecNetwork(BaseModel):
    # use gt to train in teacher forcing manner
    def __init__(self,
                 cfg,
                 vq_dim = 1024,
                 latent_dim = 768,
                 audio_encode_dim = 768,
                 period = 30,
                 n_heads = 4,
                 max_seq_len = 600,
                 id_dim = 8,
                 feedforward_dim = 1024,
                 num_layers = 1,
                 dropout = 0.1,
                 activation = 'relu',
                 window_size = 0
                 ):
        super().__init__()
        self.cfg = cfg
        self.vq_dim = vq_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        # motion encoder
        self.vq_latent_map = nn.Linear(vq_dim, latent_dim)
        # audio encoder
        self.audio_feature_map = nn.Linear(audio_encode_dim, latent_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(latent_dim, period = period)

        # noise scheduler
        self.noise_scheduler = DDPMScheduler(
            **(cfg.MODEL).get("NOISE_SCHEDULER", dict())
        )
        # denoise scheduler
        self.scheduler = DDIMScheduler(
            **(cfg.MODEL).get("SCHEDULER", dict())
        )

        self.temporal_bias_mask = temporal_bias_mask(n_head = n_heads, max_seq_len = max_seq_len, period= period)
        self.align_bias_mask = align_bias_mask(max_seq_len)

        # transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, 
                                                   nhead=n_heads, 
                                                   dim_feedforward=feedforward_dim,
                                                   dropout=dropout, 
                                                   activation=activation, 
                                                   batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        self.denoiser = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim),
        )

        self.latent_vq_map = nn.Linear(latent_dim, vq_dim)
        nn.init.xavier_normal_(self.latent_vq_map.weight)
        nn.init.xavier_normal_(self.vq_latent_map.weight)

        self.time_emb = TimestepEmbedder(hidden_size=latent_dim)
        # Initialize timestep embedding MLP
        # nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # identity embedding
        self.obj_vector = nn.Embedding(id_dim, latent_dim)

    def condition_forward(self,
                          x_past,
                          hidden_state,
                          obj_emb):
        batch_size, seq_len, _ = hidden_state.shape
        device = hidden_state.device

        tgt_mask = self.temporal_bias_mask.repeat(batch_size,1,1).to(device=device) #[batch*num_head,vert_len,vert_len]
        memory_mask = self.align_bias_mask.to(device=device) # [vert_len, seq_len]

        past_latents = self.vq_latent_map(x_past) + obj_emb
        past_latents = self.PPE(past_latents)
        vert_len = past_latents.shape[1]
        
        # use past latents to give a condition
        conditions = self.transformer_decoder(
            past_latents,
            hidden_state, 
            tgt_mask=tgt_mask[:,:vert_len, :vert_len], 
            memory_mask=memory_mask[:vert_len, :seq_len]
        ) # [batch, vert_len, latent_dim]

        return conditions

    def forward(self,
                target,
                hidden_state,
                id,
                teacher_forcing = True
                ):
        """
        Perform the forward process with diffusion head 
        Params:
            vertice (torch.Tensor): [batch_size, vert_len, vert_dim], the grount truth vertices
            template (torch.Tensor): [batch_size, 1, vert_dim], the vertice template
            hidden_state (torch.Tensor): [batch_size, seq_len, latent_dim], the audio feature
            id (torch.Tensor): [batch_size, id_dim], the id of the subject, ont-hot
            tgt_mask (torch.Tensor): [batch_size, id_dim], self-attention mask
            memory_mask (torch.Tensor): [batch_size, id_dim], cross-modal-attention mask
        """
        batch_size, vert_len, latent_dim = target.shape
        device = target.device

        # style embedding and hidden state projection
        obj_embedding = self.obj_vector(torch.argmax(id, dim=1)).unsqueeze(1) # obj_embedding.shape = [batch, 1, latent_dim]
        hidden_state = self.audio_feature_map(hidden_state) # hidden_state.shape = [batch, seq_len=vert_len, latent_dim]

        x_start = target[:,1:]
        x_past = target[:,:-1]
        
        # use conditions to guide denoising
        conditions = self.condition_forward(x_past, hidden_state, obj_embedding)

        # sample and add noise
        noise = torch.randn_like(x_start)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device = device
        )
        x_t = self.noise_scheduler.add_noise(
            x_start,
            noise,
            timesteps
        )

        t_emb = self.time_emb(timesteps).unsqueeze(1)
        conditions = conditions + t_emb
        x_t_latent = self.vq_latent_map(x_t)

        # a simple mlp diffusion head
        x_pred = self.denoiser(torch.cat((x_t_latent, conditions), dim=-1))
        x_pred = self.latent_vq_map(x_pred)

        diff_loss = mean_flat((x_pred - x_start) ** 2)

        return diff_loss, x_pred
    
    # generate from white noise during validation
    def sample(self,
               latents,
               hidden_state,
               id,
               vertice = None):
        batch_size, seq_len, latent_dim = hidden_state.shape
        device = hidden_state.device

        # style embedding and hidden state projection
        obj_embedding = self.obj_vector(torch.argmax(id, dim=1)).unsqueeze(1) # obj_embedding.shape = [batch, 1, latent_dim]
        hidden_state = self.audio_feature_map(hidden_state) # hidden_state.shape = [batch, seq_len=vert_len, latent_dim]
        
        x_start = latents[:,0].unsqueeze(1)

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.MODEL.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device, non_blocking=True)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.MODEL.eta

        for i in range(seq_len):
            if i == 0: 
                past_latents = x_start

            conditions = self.condition_forward(past_latents, hidden_state, obj_embedding)
            past_latents_len = past_latents.shape[1]
            # sample a white noise
            future_latents = torch.randn(
                size = (batch_size, past_latents_len, self.vq_dim),
                dtype = torch.float,
                device = device
            )

            # scale init noise by sigma
            future_latents = future_latents * self.scheduler.init_noise_sigma

            # denoising steps
            for j, t in enumerate(timesteps):
                future_xs = self.vq_latent_map(future_latents)

                t_emb = self.time_emb(t[None,...]).unsqueeze(1)
                curr_conditions = conditions + t_emb
                
                diffusion_output = self.denoiser(torch.cat((future_xs, curr_conditions), dim=-1))

                latents_output = self.latent_vq_map(diffusion_output)

                future_latents = self.scheduler.step(latents_output, t, future_latents, **extra_step_kwargs).prev_sample

            new_output = future_latents[:,-1,:].unsqueeze(1)
            past_latents = torch.cat((past_latents, new_output), 1)

        gt_latents = latents[:,1:]
        diff_loss = mean_flat((future_latents - gt_latents) ** 2)

        return diff_loss, future_latents
    
    # generate from white noise during validation, conditions length are fixed to training horizon
    def sample_fixed(self,
               latents,
               hidden_state,
               id,
               length=60,
               vertice = None):
        batch_size, seq_len, latent_dim = hidden_state.shape
        device = hidden_state.device

        if length==0:
            length = seq_len

        # style embedding and hidden state projection
        obj_embedding = self.obj_vector(torch.argmax(id, dim=1)).unsqueeze(1) # obj_embedding.shape = [batch, 1, latent_dim]
        hidden_state = self.audio_feature_map(hidden_state) # hidden_state.shape = [batch, seq_len=vert_len, latent_dim]
        
        x_start = latents[:,0].unsqueeze(1)

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.MODEL.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device, non_blocking=True)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.MODEL.eta

        past_latents = torch.zeros(batch_size, seq_len + 1, self.vq_dim, device=device)
        for i in range(seq_len):
            if i == 0: 
                past_latents[:, 0] = x_start

            past_latents_len = min(i+1,length)

            conditions = self.condition_forward(past_latents[:,i-past_latents_len+1:i+1], hidden_state[:,i-past_latents_len+1:i+1], obj_embedding)
            
            # sample a white noise
            future_latents = torch.randn(
                size = (batch_size, past_latents_len, self.vq_dim),
                dtype = torch.float,
                device = device
            )

            # scale init noise by sigma
            future_latents = future_latents * self.scheduler.init_noise_sigma

            # denoising steps
            for j, t in enumerate(timesteps):
                future_xs = self.vq_latent_map(future_latents)

                t_emb = self.time_emb(t[None,...]).unsqueeze(1)
                curr_conditions = conditions + t_emb
                
                diffusion_output = self.denoiser(torch.cat((future_xs, curr_conditions), dim=-1))

                latents_output = self.latent_vq_map(diffusion_output)

                future_latents = self.scheduler.step(latents_output, t, future_latents, **extra_step_kwargs).prev_sample

            # update past latents[t-window_size:t+1], fully use future latents
            past_latents[:, i+2-past_latents_len:i+2] = future_latents

        gt_latents = latents[:,1:]
        pred_latents = past_latents[:,1:]

        diff_loss = mean_flat((pred_latents - gt_latents) ** 2)

        return diff_loss, pred_latents