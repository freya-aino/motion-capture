import time
import torch as T
import torch.nn as nn
import pytorch_lightning as pl

from .convolution.backbones import Backbone
from .convolution.necks import UpsampleCrossAttentionrNeck
from .convolution.heads import SelfAttentionHead
from .SAM.sam import SAM


class UpsampleCrossAttentionNetwork(pl.LightningModule):
    
    def __init__(
        self, 
        output_size,
        output_length,
        backbone_output_size,
        neck_output_size,
        depth_multiple = 0.33,
        head_latent_size = 1024,
        neck_latent_size = 1024):
        
        super().__init__()
        
        self.backbone = Backbone(output_channels=backbone_output_size, depth_multiple=depth_multiple)
        self.neck = UpsampleCrossAttentionrNeck(output_size=neck_output_size, latent_size=neck_latent_size, depth_multiple=depth_multiple)
        self.head = SelfAttentionHead(input_size=neck_output_size, output_size=output_size, output_length=output_length, latent_size=head_latent_size)
        
    def forward(self, x: T.Tensor):
        o = self.backbone(x)
        o = self.neck(*o)
        o = self.head(o)
        return o
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch["image"], batch["keypoints"]
        opt = self.optimizers()
        
        o = self(x)
        loss = T.nn.functional.mse_loss(o, y)
        self.log("train_loss_pre_opt", loss)
        
        loss.backward()
        opt.first_step(zero_grad=True)
        
        o = self(x)
        loss = T.nn.functional.mse_loss(o, y)
        self.log("train_loss_post_opt", loss)
        
        loss.backward()
        opt.second_step(zero_grad=True)
        
    
    def configure_optimizers(self):
        
        return SAM(
            params = self.parameters(),
            base_optimizer = T.optim.SGD,
            lr = 0.1,
            momentum = 0.9
        )
        