import math
import torch as T
import torch.nn as nn

from ..transformer.core import TransformerEncoderBlock
from motion_capture.core.torchhelpers import positional_embedding


class SimpleTransformerHead(nn.Module):
    def __init__(
        self, 
        input_size,
        output_size,
        output_length,
        width = 1024,
        width_multiple = 0.25):
        
        super(type(self), self).__init__()
        
        self.latent_size = int(width * width_multiple)
        
        self.output_length = output_length
        
        self.input_1d_conv = nn.Sequential(
            nn.Conv1d(input_size, self.latent_size, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm1d(self.latent_size)
        )
        
        self.positional_embedding = nn.Parameter(positional_embedding(20*20, self.latent_size), requires_grad=False)
        
        self.self_attention = TransformerEncoderBlock(self.latent_size, output_size)
        
        self.internal_state = nn.Parameter(T.rand(output_length, self.latent_size, dtype=T.float32), requires_grad=True)
        
    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.input_1d_conv(x).permute(0, 2, 1)
        x = T.cat([x, self.internal_state.expand(x.shape[0], -1, -1)], 1)
        x = self.self_attention(x)
        return x[:, -self.output_length:, :]


class CodebookTransformerHead(nn.Module):
    def __init__(
        self, 
        input_size,
        output_size,
        width_multiple = 0.25):
        
        super(type(self), self).__init__()
        raise NotImplementedError()


class CascadedTransformerHead(nn.Module):
    def __init__(
        self, 
        input_size,
        output_size,
        width = 1024,
        width_multiple = 0.25):
        super(type(self), self).__init__()
        
        self.latent_size = int(width * width_multiple)
        
        self.input_1d_conv = nn.Conv1d(input_size, self.latent_size, kernel_size=1, stride=1, padding=0, groups=1)
        self.input_1d_conv_memory = nn.Conv1d(input_size, self.latent_size, kernel_size=1, stride=1, padding=0, groups=1)
        
        self.positional_embedding = nn.Parameter(positional_embedding(20*20, self.latent_size), requires_grad=False)
        
        self.forward_encoder = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.SiLU(),
            nn.BatchNorm1d(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.SiLU(),
            nn.BatchNorm1d(self.latent_size)
        )
        
        self.R = nn.Parameter(T.rand(output_size, dtype=T.float32), requires_grad=True)
        
    def forward(self, x: T.Tensor, corrections: int = 3) -> T.Tensor:
        
        raise NotImplementedError
        
        M = self.input_1d_conv_memory(x).permute(0, 2, 1)
        R = self.R
        X = self.input_1d_conv(x).permute(0, 2, 1)
        
        for _ in range(corrections):
            
            q = X + self.positional_embedding.expand(X.shape[0], -1, -1)
            k = X + self.positional_embedding.expand(X.shape[0], -1, -1)
            v = X
            
            Q = X + nn.functional.scaled_dot_product_attention(q, k, v)
            
            print(Q.shape, M.shape, R.shape)
            
            y = nn.functional.scaled_dot_product_attention(Q, M, R)
            y = y + Q
            y = self.forward_encoder(y)
            
            print(y.shape)
            
        
        return out


class DeformableAttentionHead(nn.Module):
    def __init__(
        self, 
        input_size,
        output_size,
        width = 1024,
        width_multiple = 0.25):
        
        raise NotImplementedError()

        super(type(self), self).__init__()
        
        self.latent_size = int((width * width_multiple))
        
        self.latent_size = int(self.latent_size * width_multiple)
        
        self.positional_embedding = nn.Parameter(positional_embedding(20*20, input_size), requires_grad=False)
        
        self.deformable_attn = DeformableAttention(
            dim = input_size,
            dim_head = input_size // 8,  # dimension per head
            heads = 8,
            dropout = 0.,
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = None,        # number of offset groups, should be multiple of heads
            offset_kernel_size = 6,
        )
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])))
        out = self.deformable_attn(x)
        return out
