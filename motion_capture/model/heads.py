from importlib import import_module
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .convolution import ConvBlock, C2f, Detection, SPPF
from .transformer import PyramidTransformer, LL_LM_Attention


class YOLOv8Head(nn.Module):
    # model head from: https://blog.roboflow.com/whats-new-in-yolov8/
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        output_lenght: int,
        num_classes: int,
        depth: int = 1,
        ):
        super(type(self), self).__init__()
        
        assert depth >= 1, "depth must be at least 1"
        
        l0 = input_channels
        l1 = input_channels // 2
        l2 = input_channels // 4
        
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.c2f_1 = C2f(l0 + l1, l1, kernel_size=1, n=depth, shortcut=False)
        self.c2f_2 = C2f(l1 + l2, l2, kernel_size=1, n=depth, shortcut=False)
        
        self.conv_1 = ConvBlock(l2, l2, kernel_size=3, stride=2, padding=1)
        self.c2f_3 = C2f(l2 + l1, l1, kernel_size=1, n=depth, shortcut=False)
        
        self.conv_2 = ConvBlock(l1, l1, kernel_size=3, stride=2, padding=1)
        self.c2f_4 = C2f(l1 + l0, l1, kernel_size=1, n=depth, shortcut=False)
        
        self.det_1 = Detection(l2, output_lenght, num_classes)
        self.det_2 = Detection(l1, output_lenght, num_classes)
        self.det_3 = Detection(l1, output_lenght, num_classes)
        
    def forward(self, x: list):
        x1, x2, x3 = x
        
        z = T.cat([self.upsample_x2(x3), x2], 1)
        z = self.c2f_1(z)
        
        y1 = T.cat([self.upsample_x2(z), x1], 1)
        y1 = self.c2f_2(y1)
        
        y2 = T.cat([self.conv_1(y1), z], 1)
        y2 = self.c2f_3(y2)
        
        y3 = T.cat([self.conv_2(y2), x3], 1)
        y3 = self.c2f_4(y3)
        
        y1 = self.det_1(y1)
        y2 = self.det_2(y2)
        y3 = self.det_3(y3)
        
        return y1, y2, y3
    
    def compute_loss(self, y_pred, y):
        loss_fn = T.nn.functional.smooth_l1_loss if self.continuous_output else T.nn.functional.cross_entropy
        return loss_fn(y_pred, y)

class PyramidTransformerHead(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(
        self, 
        input_dims: int,
        input_length: int,
        output_dims: int,
        output_length: int,
        num_heads: int
        ):
        super().__init__()
        
        self.output_shape = (output_length, output_dims)
        
        
        self.upsample_2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_dims, input_dims // 2, 1, 1)
        )
        self.pyramid_transformer_2 = PyramidTransformer(input_dims // 2, input_length * 4, num_heads)
        
        self.upsample_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_dims // 2, input_dims // 4, 1, 1)
        )
        self.pyramid_transformer_1 = PyramidTransformer(input_dims // 4, input_length * 16, num_heads)
        
        
        self.memory_encoder_3 = nn.Sequential(C2f(input_dims, output_dims, 1, shortcut=True), nn.Flatten(2))
        self.memory_encoder_2 = nn.Sequential(C2f(input_dims // 2, output_dims, 1, shortcut=True), nn.Flatten(2))
        self.memory_encoder_1 = nn.Sequential(C2f(input_dims // 4, output_dims, 1, shortcut=True), nn.Flatten(2))
        
        self.ll_lm_attention_3 = LL_LM_Attention(output_dims, output_length, input_length, num_heads)
        self.ll_lm_attention_2 = LL_LM_Attention(output_dims, output_length, input_length * 4, num_heads)
        self.ll_lm_attention_1 = LL_LM_Attention(output_dims, output_length, input_length * 16, num_heads)
        
        self.initial_predictor = nn.Sequential(
            SPPF(input_dims, input_dims),
            C2f(input_dims, input_dims, 2, shortcut=True),
            C2f(input_dims, output_dims * output_length, 2, shortcut=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
    def forward(self, x):
        
        z_1, z_2, z_3 = x
        
        memory_3 = z_3
        
        memory_2 = self.pyramid_transformer_2(
            feature_map = z_2, 
            memory = self.upsample_2(memory_3)
        )
        
        memory_1 = self.pyramid_transformer_1(
            feature_map = z_1, 
            memory = self.upsample_1(memory_2)
        )
        
        enc_mem_3 = self.memory_encoder_3(memory_3).permute(0, 2, 1)
        enc_mem_2 = self.memory_encoder_2(memory_2).permute(0, 2, 1)
        enc_mem_1 = self.memory_encoder_1(memory_1).permute(0, 2, 1)
        
        
        pred_3 = self.initial_predictor(memory_3).reshape(-1, *self.output_shape)
        pred_2 = self.ll_lm_attention_3(pred_3, enc_mem_3) + pred_3
        pred_1 = self.ll_lm_attention_2(pred_2, enc_mem_2) + pred_2
        pred_0 = self.ll_lm_attention_1(pred_1, enc_mem_1) + pred_1
        
        return pred_3, pred_2, pred_1, pred_0
    
    def compute_loss(self, y_pred, y, loss_fn = F.l1_loss, **loss_fn_kwargs):
        # compute residual loss for each of the predictions
        return [loss_fn(y_p, y, **loss_fn_kwargs) for y_p in y_pred]