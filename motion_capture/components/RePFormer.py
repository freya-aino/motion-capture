import torch as T
import torch.nn as nn
import torch.nn.functional as F
import motion_capture.components.convolution as conv

from motion_capture.utils.torchhelpers import positionalencoding1d

# def make_even(x: int) -> int:
#     return x if x % 2 == 0 else x + 1

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
