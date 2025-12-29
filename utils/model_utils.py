import torch
import math

# Borrowed from https://github.com/EvelynFan/FaceFormer/blob/dfaea81983665b22b99af336a80574208cfcc099/faceformer.py#L10
def temporal_bias_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def align_bias_mask(max_seq_len):
    mask = torch.ones(max_seq_len, max_seq_len)
    mask = mask.masked_fill(torch.eye(max_seq_len) == 1, 0)
    return mask==1

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def mean_flat(tensor):
    """
    Take the mean over all dimensions.
    """
    return tensor.mean(dim=list(range(0, len(tensor.shape))))