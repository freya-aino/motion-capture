import time
import torch as T
import torch.nn as nn

from .convolution import ConvBlock, C2f, SPPF

class Backbone(nn.Module):
    def __init__(
        self, 
        output_channels: int,
        depth: int = 1,
        input_channels: int = 3):
        super(type(self), self).__init__()
        
        assert output_channels / 16 >= 1, "output_channels must be at least be divisible by 16"
        assert depth >= 1, "depth_multiple must be at least 1"
        
        size_4 = output_channels 
        size_3 = output_channels // 2
        size_2 = output_channels // 4
        size_1 = output_channels // 8
        size_0 = output_channels // 16
        
        self.conv1 = nn.Sequential(
            ConvBlock(input_channels, size_0, kernel_size=3, stride=2, padding=1),
            ConvBlock(size_0, size_1, kernel_size=3, stride=2, padding=1),
            C2f(size_1, size_1, kernel_size=1, n=depth, shortcut=True),
            ConvBlock(size_1, size_2, kernel_size=3, stride=2, padding=1),
            C2f(size_2, size_2, kernel_size=1, n=depth, shortcut=True)
        )
        
        self.conv2 = nn.Sequential(
            ConvBlock(size_2, size_3, kernel_size=3, stride=2, padding=1),
            C2f(size_3, size_3, kernel_size=1, n=2*depth, shortcut=True),
        )
        
        self.conv3 = nn.Sequential(
            ConvBlock(size_3, size_4, kernel_size=3, stride=2, padding=1),
            C2f(size_4, size_4, kernel_size=1, n=depth, shortcut=True),
            SPPF(size_4, size_4)
        )
        
        self.batch_norm_x1 = nn.BatchNorm2d(size_2)
        self.batch_norm_x2 = nn.BatchNorm2d(size_3)
        self.batch_norm_x3 = nn.BatchNorm2d(size_4)
    
    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x1 = self.batch_norm_x1(x1)
        x2 = self.batch_norm_x2(x2)
        x3 = self.batch_norm_x3(x3)
        
        return x1, x2, x3
