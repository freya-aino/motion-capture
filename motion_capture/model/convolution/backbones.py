import time
import torch as T
import torch.nn as nn

from .core import ConvBlock, C2f, SPPF

class Backbone(nn.Module):
    def __init__(self, depth_multiple = 0.33, width_multiple = 0.25):
        
        super(type(self), self).__init__()
        
        scaled_64 = int(64 * width_multiple)
        scaled_128 = int(128 * width_multiple)
        scaled_256 = int(256 * width_multiple)
        scaled_512 = int(512 * width_multiple)
        scaled_1024 = int(1024 * width_multiple)
        
        self.conv1 = nn.Sequential(
            ConvBlock(3, scaled_64, kernel_size=3, stride=2, padding=1),
            ConvBlock(scaled_64, scaled_128, kernel_size=3, stride=2, padding=1),
            C2f(scaled_128, scaled_128, kernel_size=1, n=int(3*depth_multiple), shortcut=True),
            ConvBlock(scaled_128, scaled_256, kernel_size=3, stride=2, padding=1),
            C2f(scaled_256, scaled_256, kernel_size=1, n=int(6*depth_multiple), shortcut=True)
        )
        
        self.conv2 = nn.Sequential(
            ConvBlock(scaled_256, scaled_512, kernel_size=3, stride=2, padding=1),
            C2f(scaled_512, scaled_512, kernel_size=1, n=int(6*depth_multiple), shortcut=True),
        )
        
        self.conv3 = nn.Sequential(
            ConvBlock(scaled_512, scaled_1024, kernel_size=3, stride=2, padding=1),
            C2f(scaled_1024, scaled_1024, kernel_size=1, n=int(3*depth_multiple), shortcut=True),
            SPPF(scaled_1024, scaled_1024)
        )
    
    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1, x2, x3

