import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange
from torch.nn import functional as F


class Norm(nn.Module):
  """ Norm Layer """

  def __init__(self, fn, size):
    super().__init__()
    self.norm = nn.LayerNorm(size, eps=1e-5)
    self.fn = fn

  def forward(self, x_data):
    if type(x_data) is dict:
        x_norm = self.fn({'x_a':x_data['x_a'], 'x_b':self.norm(x_data['x_b'])})
        return x_norm
    else:
        x, mask_info = x_data
        x_norm, _ = self.fn((self.norm(x), mask_info))
        return (x_norm, mask_info)

class Residual(nn.Module):
  """ Residual Layer """

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x_data):
    if type(x_data) is dict:
        x_resid = self.fn(x_data)['x_b']
        return {'x_a':x_data['x_a'], 'x_b':x_resid+x_data['x_b']}
    else:
        x, mask_info = x_data
        x_resid, _ = self.fn(x_data)
        return (x_resid + x, mask_info)


class MLP(nn.Module):
  """ MLP Layer """

  def __init__(self, in_dim, out_dim, hidden_dim):
    super().__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.activation = nn.GELU()
    self.l2 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x_data):
    if type(x_data) is dict:
        out = self.l2(self.activation(self.l1(x_data['x_b'])))
        return {'x_a':x_data['x_a'], 'x_b':out}
    else:
        x, mask_info = x_data
        out = self.l2(self.activation(self.l1(x)))
        return (out, mask_info)


class CrossModalAttention(nn.Module):
  """ Cross Modal Attention Layer
  Given 2 modalities (a, b), computes the K,V from modality b and Q from
  modality a.
  """

  def __init__(self, in_dim, dim, heads=8, in_dim2=None):
    super().__init__()
    self.heads = heads
    self.scale = dim**-0.5

    if in_dim2 is not None:
        self.to_kv = nn.Linear(in_dim2, in_dim2 * 2, bias=False)
    else:
        self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
    self.to_q = nn.Linear(in_dim, dim, bias=False)
    if in_dim2 is not None:
        dim2 = int((in_dim + in_dim2*2) / 3)
    else:
        dim2 = dim
    self.to_out = nn.Linear(dim2, dim)

    self.rearrange_qkv = Rearrange(
        "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
    self.rearrange_out = Rearrange("b h n d -> b n (h d)")

  def forward(self, x_data):
    x_a = x_data['x_a']
    x_b = x_data['x_b']

    kv = self.to_kv(x_b)
    q = self.to_q(x_a)

    qkv = torch.cat((q, kv), dim=-1)
    qkv = self.rearrange_qkv(qkv)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    attn = F.softmax(dots, dim=-1)

    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = self.rearrange_out(out)
    out = self.to_out(out)
    return {'x_a':x_a, 'x_b':out}


class Attention(nn.Module):
  """ Attention Layer """

  def __init__(self, in_dim, dim, heads=8):
    super().__init__()
    self.heads = heads
    self.scale = dim**-0.5

    self.to_qkv = nn.Linear(in_dim, dim * 3, bias=False)
    self.to_out = nn.Linear(dim, dim)

    self.rearrange_qkv = Rearrange(
        "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
    self.rearrange_out = Rearrange("b h n d -> b n (h d)")

  def forward(self, x_data):
    x, mask_info = x_data
    max_mask = mask_info['max_mask']
    mask = mask_info['mask']
    #
    qkv = self.to_qkv(x)
    qkv = self.rearrange_qkv(qkv)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    if max_mask is not None:
        dots[:,:,:max_mask,:max_mask] = \
            dots[:,:,:max_mask,:max_mask].masked_fill(mask == 0., float('-inf'))

    attn = F.softmax(dots, dim=-1)

    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = self.rearrange_out(out)
    out = self.to_out(out)
    return (out, mask_info)

class Transformer(nn.Module):
  """ Transformer class
  Parameters
  ----------
  cross_modal : bool
    if true, uses cross-modal attention layers, else is the vanilla Transformer
  in_dim2 : int
    specifies the feature size of the second modality if using cross_modal
  """

  def __init__(self,
               in_size=50,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               cross_modal=False,
               in_dim2=None):
    super().__init__()
    blocks = []
    attn = False

    self.cross_modal = cross_modal
    if cross_modal:
      for i in range(num_hidden_layers):
        blocks.extend([
            Residual(Norm(CrossModalAttention(in_size, hidden_size,
                                              heads=num_attention_heads,
                                              in_dim2=in_dim2), hidden_size)),
            Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                              hidden_size))
        ])
    else:
      for i in range(num_hidden_layers):
        blocks.extend([
            Residual(Norm(Attention(in_size, hidden_size,
                                    heads=num_attention_heads), hidden_size)),
            Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                              hidden_size))
        ])
    self.net = torch.nn.Sequential(*blocks)

  def forward(self, x_data):
    if self.cross_modal:
      assert type(x_data) is dict
      x_data = self.net(x_data)
      x = x_data['x_b']
    else:
      x, mask_info = x_data
      x, _ = self.net((x, mask_info))
    return x