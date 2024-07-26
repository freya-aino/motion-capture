import torch as T
import torch.nn as nn


# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, depth):
#         super(type(self), self).__init__()
        
#         assert depth > 0, "depth must be greater than 0"
        
#         self.LN1 = nn.LayerNorm(normalized_shape=input_dim, eps=0.00001, elementwise_affine=True)
#         self.LN2 = nn.LayerNorm(normalized_shape=output_dim, eps=0.00001, elementwise_affine=True)
        
#         self.MHA = nn.MultiheadAttention(
#             embed_dim=input_dim,
#             num_heads=1,
#             dropout=0,
#             bias=True,
#             add_bias_kv=False,
#             add_zero_attn=False,
#             batch_first=True)
        
#         encoder_dims = [int(input_dim + (output_dim - input_dim) * (n / depth)) for n in range(depth + 1)]
#         self.MLP = nn.Sequential(*[
#             nn.Sequential(
#                 nn.Linear(in_features=d1, out_features=d2, bias=True),
#                 nn.SiLU()
#             ) for (d1, d2) in zip(encoder_dims[:-1], encoder_dims[1:])
#             ]
#         )
        
#         self.residual_1 = nn.Sequential(nn.Linear(input_dim, input_dim), nn.SiLU())
#         self.residual_2 = nn.Sequential(nn.Linear(input_dim, output_dim), nn.SiLU())
        
#     def forward(self, x: T.Tensor, attention_mask = None, return_attention_matrix: bool = False):
#         if attention_mask is not None:
#             attention_mask = attention_mask.expand(x.shape[0], -1, -1)
        
#         y1, attention_matrix = self.MHA(x, x, x, attn_mask=attention_mask)
#         y1 = self.LN1(y1 + self.residual_1(x))
        
#         y2 = self.MLP(y1)
#         y2 = self.LN2(y2 + self.residual_2(y1))
        
#         if return_attention_matrix:
#             return y2, attention_matrix
#         return y2


# class TransformerDecoderBlock(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int):
#         super(type(self), self).__init__()
        
#         self.LN1 = nn.LayerNorm(normalized_shape=input_dim, eps=0.00001, elementwise_affine=True)
#         self.LN2 = nn.LayerNorm(normalized_shape=input_dim, eps=0.00001, elementwise_affine=True)
#         self.LN3 = nn.LayerNorm(normalized_shape=input_dim, eps=0.00001, elementwise_affine=True)
        
#         self.MHA1 = nn.MultiheadAttention(
#             embed_dim=input_dim,
#             num_heads=1,
#             dropout=0,
#             bias=True,
#             add_bias_kv=False,
#             add_zero_attn=False,
#             batch_first=True)
        
#         self.MHA2 = nn.MultiheadAttention(
#             embed_dim=input_dim,
#             num_heads=1,
#             dropout=0,
#             bias=True,
#             add_bias_kv=False,
#             add_zero_attn=False,
#             batch_first=True)
        
#         self.MLP = nn.Sequential(
#             nn.Linear(in_features=input_dim, out_features=(input_dim + output_dim) // 2, bias=True),
#             nn.SiLU(),
#             nn.Linear(in_features=(input_dim + output_dim) // 2, out_features=output_dim, bias=True),
#             nn.SiLU(),
#         )
        
#         self.residual_1 = nn.Sequential(nn.Linear(input_dim, output_dim), nn.SiLU())
        
        
#     def forward(self, x: T.Tensor, memory: T.Tensor, attention_mask = None, return_attention_matrix: bool = False):
#         if attention_mask is not None:
#             attention_mask = attention_mask.expand(x.shape[0], -1, -1)
        
        
#         y1 = self.MHA1(x, x, x)[0]
#         y1 = self.LN1(y1 + self.residual_1(x))
        
#         y2 = self.MHA2(y1, memory, memory, attn_mask=attention_mask)[0]
        
        
        
        
#         x, attention_matrix1 = self.MHA1(x, x, x)
#         x = self.LN2(x)
#         x, attention_matrix2 = self.MHA2(x, memory, memory)
#         x = self.MLP(self.LN3(x))
#         x = T.add(residual, x)
        
#         if return_attention_matrix:
#             return x, attention_matrix1, attention_matrix2
#         return x
    
