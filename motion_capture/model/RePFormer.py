import torch as T
import torch.nn as nn
import torch.nn.functional as F
import motion_capture.model.convolution as conv

from motion_capture.core.torchhelpers import positionalencoding1d

# def make_even(x: int) -> int:
#     return x if x % 2 == 0 else x + 1

# class AttentionBlock(nn.Module):
#     def __init__(self, input_dims: int, num_heads: int):
#         super().__init__()
        
#         self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dims, num_heads=num_heads, batch_first=True)
#         self.attention_norm_1 = nn.LayerNorm(input_dims)
#         self.attention_ff = nn.Sequential(nn.Linear(input_dims, input_dims), nn.SiLU())
#         self.attention_norm_2 = nn.LayerNorm(input_dims)
        
#     def forward(self, q, k, v):
#         z_hat, _ = self.multi_head_attention(query = q, key = k, value = v)
#         z_hat = self.attention_norm_1(z_hat + q)
#         y = self.attention_norm_2(self.attention_ff(z_hat) + z_hat)
#         return y

# class PyramidTransformer(nn.Module):
#     def __init__(self, input_dims: int, input_length: int, num_heads: int):
#         super().__init__()
        
#         self.attention_block = AttentionBlock(input_dims, num_heads)
#         self.positional_encoding = nn.Parameter(positionalencoding1d(make_even(input_length), make_even(input_dims))[:input_dims, :input_length], requires_grad=False)
        
#     def forward(self, feature_map, memory):
        
#         in_shape = feature_map.shape
        
#         v_hat = (feature_map * memory).flatten(2)
#         pos_enc = self.positional_encoding.expand(feature_map.shape[0], -1, -1)
#         feature_map = feature_map.flatten(2)
        
#         y = self.attention_block(
#             q = (feature_map + pos_enc).permute(0, 2, 1),
#             k = (v_hat + pos_enc).permute(0, 2, 1),
#             v = v_hat.permute(0, 2, 1)
#         )
        
#         return y.permute(0, 2, 1).reshape(in_shape)

# class LL_LM_Attention(nn.Module):
#     def __init__(self, input_dims: int, query_length: int, memory_length: int, num_heads: int):
#         super().__init__()
        
#         self.query_positional_encoding = nn.Parameter(positionalencoding1d(make_even(query_length), make_even(input_dims))[:input_dims, :query_length].T, requires_grad=False)
#         self.memory_positional_encoding = nn.Parameter(positionalencoding1d(make_even(memory_length), make_even(input_dims))[:input_dims, :memory_length].T, requires_grad=False)
        
#         self.ll_attention_block = AttentionBlock(input_dims, 1)
#         self.lm_attention_block = AttentionBlock(input_dims, 1)
        
#     def forward(self, queries, memory):
        
#         qk = queries + self.query_positional_encoding.expand(queries.shape[0], -1, -1)
        
#         queries = self.ll_attention_block(qk, qk, queries)
        
#         lm_q = queries + self.query_positional_encoding.expand(queries.shape[0], -1, -1)
#         lm_k = memory + self.memory_positional_encoding.expand(memory.shape[0], -1, -1)
        
#         queries = self.lm_attention_block(lm_q, lm_k, memory)
        
#         return queries

# class PyramidTransformerHead(nn.Module):
#     # based on: https://arxiv.org/pdf/2207.03917.pdf
#     def __init__(
#         self, 
#         input_dims: int,
#         input_length: int,
#         output_dims: int,
#         output_length: int,
#         num_heads: int
#         ):
#         super().__init__()
        
#         self.output_shape = (output_length, output_dims)
        
        
#         self.upsample_2 = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.Conv2d(input_dims, input_dims // 2, 1, 1)
#         )
#         self.pyramid_transformer_2 = PyramidTransformer(input_dims // 2, input_length * 4, num_heads)
        
#         self.upsample_1 = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.Conv2d(input_dims // 2, input_dims // 4, 1, 1)
#         )
#         self.pyramid_transformer_1 = PyramidTransformer(input_dims // 4, input_length * 16, num_heads)
        
        
#         self.memory_encoder_3 = nn.Sequential(conv.C2f(input_dims, output_dims, 1, shortcut=True), nn.Flatten(2))
#         self.memory_encoder_2 = nn.Sequential(conv.C2f(input_dims // 2, output_dims, 1, shortcut=True), nn.Flatten(2))
#         self.memory_encoder_1 = nn.Sequential(conv.C2f(input_dims // 4, output_dims, 1, shortcut=True), nn.Flatten(2))
        
#         self.ll_lm_attention_3 = LL_LM_Attention(output_dims, output_length, input_length, num_heads)
#         self.ll_lm_attention_2 = LL_LM_Attention(output_dims, output_length, input_length * 4, num_heads)
#         self.ll_lm_attention_1 = LL_LM_Attention(output_dims, output_length, input_length * 16, num_heads)
        
#         self.initial_predictor = nn.Sequential(
#             conv.SPPF(input_dims, input_dims),
#             conv.C2f(input_dims, input_dims, 2, shortcut=True),
#             conv.C2f(input_dims, output_dims * output_length, 2, shortcut=False),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(1)
#         )
        
#     def forward(self, x):
        
#         z_1, z_2, z_3 = x
        
#         memory_3 = z_3
        
#         memory_2 = self.pyramid_transformer_2(
#             feature_map = z_2, 
#             memory = self.upsample_2(memory_3)
#         )
        
#         memory_1 = self.pyramid_transformer_1(
#             feature_map = z_1, 
#             memory = self.upsample_1(memory_2)
#         )
        
#         enc_mem_3 = self.memory_encoder_3(memory_3).permute(0, 2, 1)
#         enc_mem_2 = self.memory_encoder_2(memory_2).permute(0, 2, 1)
#         enc_mem_1 = self.memory_encoder_1(memory_1).permute(0, 2, 1)
        
        
#         pred_3 = self.initial_predictor(memory_3).reshape(-1, *self.output_shape)
#         pred_2 = self.ll_lm_attention_3(pred_3, enc_mem_3) + pred_3
#         pred_1 = self.ll_lm_attention_2(pred_2, enc_mem_2) + pred_2
#         pred_0 = self.ll_lm_attention_1(pred_1, enc_mem_1) + pred_1
        
#         return pred_3, pred_2, pred_1, pred_0
    
#     def compute_loss(self, y_pred, y, loss_fn = F.l1_loss):
#         # compute residual loss for each of the predictions
#         return [loss_fn(y_p, y) for y_p in y_pred]