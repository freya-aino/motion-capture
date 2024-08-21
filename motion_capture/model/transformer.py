import torch as T
import torch.nn as nn

from motion_capture.core.torchhelpers import positionalencoding1d

def make_even(x: int) -> int:
    return x if x % 2 == 0 else x + 1

class AttentionBlock(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(self, input_dims: int, num_heads: int):
        super().__init__()
        
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dims, num_heads=num_heads, batch_first=True)
        self.attention_norm_1 = nn.LayerNorm(input_dims)
        self.attention_ff = nn.Sequential(nn.Linear(input_dims, input_dims), nn.SiLU())
        self.attention_norm_2 = nn.LayerNorm(input_dims)
        
    def forward(self, q, k, v):
        z_hat, _ = self.multi_head_attention(query = q, key = k, value = v)
        z_hat = self.attention_norm_1(z_hat + q)
        y = self.attention_norm_2(self.attention_ff(z_hat) + z_hat)
        return y

class PyramidTransformer(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(self, input_dims: int, input_length: int, num_heads: int):
        super().__init__()
        
        self.attention_block = AttentionBlock(input_dims, num_heads)
        self.positional_encoding = nn.Parameter(positionalencoding1d(make_even(input_length), make_even(input_dims))[:input_dims, :input_length], requires_grad=False)
        
    def forward(self, feature_map, memory):
        
        in_shape = feature_map.shape
        
        v_hat = (feature_map * memory).flatten(2)
        pos_enc = self.positional_encoding.expand(feature_map.shape[0], -1, -1)
        feature_map = feature_map.flatten(2)
        
        y = self.attention_block(
            q = (feature_map + pos_enc).permute(0, 2, 1),
            k = (v_hat + pos_enc).permute(0, 2, 1),
            v = v_hat.permute(0, 2, 1)
        )
        
        return y.permute(0, 2, 1).reshape(in_shape)

class LL_LM_Attention(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(self, input_dims: int, query_length: int, memory_length: int, num_heads: int):
        super().__init__()
        
        self.query_positional_encoding = nn.Parameter(positionalencoding1d(make_even(query_length), make_even(input_dims))[:input_dims, :query_length].T, requires_grad=False)
        self.memory_positional_encoding = nn.Parameter(positionalencoding1d(make_even(memory_length), make_even(input_dims))[:input_dims, :memory_length].T, requires_grad=False)
        
        self.ll_attention_block = AttentionBlock(input_dims, 1)
        self.lm_attention_block = AttentionBlock(input_dims, 1)
        
    def forward(self, queries, memory):
        
        qk = queries + self.query_positional_encoding.expand(queries.shape[0], -1, -1)
        
        queries = self.ll_attention_block(qk, qk, queries)
        
        lm_q = queries + self.query_positional_encoding.expand(queries.shape[0], -1, -1)
        lm_k = memory + self.memory_positional_encoding.expand(memory.shape[0], -1, -1)
        
        queries = self.lm_attention_block(lm_q, lm_k, memory)
        
        return queries


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
    
