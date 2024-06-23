import time
import torch as T
import torch.nn as nn

from .core import ConvBlock, C2f, SPPF

class Backbone(nn.Module):
    def __init__(
        self, 
        output_channels: int = 1024,
        depth_multiple: int = 1):
        super(type(self), self).__init__()
        
        assert output_channels / 8 > 1, "output_channels must be at least be divisible by 8"
        assert depth_multiple >= 1, "depth_multiple must be at least 1"
        
        size_4 = output_channels
        size_3 = int(size_4 / 2)
        size_2 = int(size_3 / 2)
        size_1 = int(size_2 / 2)
        size_0 = int(size_1 / 2)
        
        self.conv1 = nn.Sequential(
            ConvBlock(3, size_0, kernel_size=3, stride=2, padding=1),
            ConvBlock(size_0, size_1, kernel_size=3, stride=2, padding=1),
            C2f(size_1, size_1, kernel_size=1, n=int(depth_multiple), shortcut=True),
            ConvBlock(size_1, size_2, kernel_size=3, stride=2, padding=1),
            C2f(size_2, size_2, kernel_size=1, n=int(depth_multiple), shortcut=True)
        )
        
        self.conv2 = nn.Sequential(
            ConvBlock(size_2, size_3, kernel_size=3, stride=2, padding=1),
            C2f(size_3, size_3, kernel_size=1, n=int(2*depth_multiple), shortcut=True),
        )
        
        self.conv3 = nn.Sequential(
            ConvBlock(size_3, size_4, kernel_size=3, stride=2, padding=1),
            C2f(size_4, size_4, kernel_size=1, n=int(depth_multiple), shortcut=True),
            SPPF(size_4, size_4)
        )
    
    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1, x2, x3

