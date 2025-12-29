import torch
import torch.nn as nn
import math

from torch.utils.checkpoint import checkpoint

from utils.model_utils import modulate
from algorithms.models.architectures.common import TimestepEmbedder

# latent diffusion denoiser with condition z
class Latent_Denoiser(nn.Module):
    """
        Implementation of Latent Diffusion with condition z
        Params:
            target_channels: equals to dim of n_verts
            z_channels: dim of condition
            mlp_width: dim of hidden layers
            mlp_layers: network depth
            grad_checkpointing: save gpu memory, slower

    """
    def __init__(self, 
                 target_channels, 
                 z_channels, 
                 mlp_layers, 
                 mlp_width, 
                 grad_checkpointing=False):
        super().__init__()

        self.target_channels = target_channels
        self.z_channels = z_channels
        self.mlp_layers = mlp_layers
        self.mlp_width = mlp_width
        self.grad_checkpointing = grad_checkpointing

        # timestep proj
        self.time_embed = TimestepEmbedder(mlp_width)

        # residual blocks
        res_blocks = []
        for i in range(mlp_layers):
            res_blocks.append(ResBlock(
                mlp_width,
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        # final proj layer
        self.final_layer = FinalLayer(mlp_width, target_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, xt, timesteps, z):
        """
        Latent Denoising process, with guidance of condition z
        Params:
            xt: [batch, seq_len, latent_dim]
            timesteps: a 1-D batch of timesteps.[batch]
            z: condition from AR transformer. [batch, seq_len, latent_dim]
        Return:
            output : denoised result [batch, seq_len, latent_dim]
        """

        t = self.time_embed(timesteps).unsqueeze(1)
        y = t + z

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                xt = checkpoint(block, xt, y)
        else:
            for block in self.res_blocks:
                xt = block(xt, y)

        return self.final_layer(xt, y)

# Borrowed from MAR: https://github.com/LTH14/mar/tree/main
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


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
