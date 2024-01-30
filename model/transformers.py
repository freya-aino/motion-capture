import time
import torch as T
import torch.nn as nn
import torch.nn.functional as nnF



class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(type(self), self).__init__()

        self.LN1 = nn.LayerNorm(normalized_shape=input_dim, eps=0.00001, elementwise_affine=True)
        self.LN2 = nn.LayerNorm(normalized_shape=input_dim, eps=0.00001, elementwise_affine=True)
        
        self.MSA = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=1,
            dropout=0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True)
        
        self.MLP = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=(input_dim + output_dim) // 2, bias=True),
            nn.ELU(),
            nn.Linear(in_features=(input_dim + output_dim) // 2, out_features=output_dim, bias=True),
            nn.ELU(),
        )
        
        self.residual_connection = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ELU())
        
    def forward(self, x: T.Tensor):
        residual = self.residual_connection(x)
        x = self.LN1(x)
        x, attention_matrix = self.MSA(x, x, x)
        x = self.MLP(self.LN2(x))
        x = T.add(residual, x)
        return x
