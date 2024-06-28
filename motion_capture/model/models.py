import time
import torch as T
import torch.nn as nn
import pytorch_lightning as pl

from .convolution.backbones import Backbone
from .convolution.necks import UpsampleCrossAttentionrNeck
from .convolution.heads import SelfAttentionHead
from .SAM.sam import SAM
from .SAM.example.utility.bypass_bn import enable_running_stats, disable_running_stats

class UpsampleCrossAttentionNetwork(pl.LightningModule):
    
    def __init__(
        self, 
        output_size: int,
        output_length: int, 
        backbone_output_size,
        neck_output_size,
        head_latent_size,
        loss_fn = T.nn.functional.mse_loss,
        depth_multiple = 1):
        
        super().__init__()
        
        self.automatic_optimization = False
        
        self.loss_fn = loss_fn
        
        self.backbone = Backbone(output_channels=backbone_output_size, depth_multiple=depth_multiple)
        self.neck = UpsampleCrossAttentionrNeck(output_size=neck_output_size, latent_size=backbone_output_size, depth_multiple=depth_multiple)
        self.head = SelfAttentionHead(input_size=neck_output_size, output_size=output_size, output_length=output_length, latent_size=head_latent_size)
        
    def forward(self, x: T.Tensor):
        resnet_residuals = self.backbone(x)
        neck_out = self.neck(*resnet_residuals)
        head_out = self.head(neck_out)
        return head_out
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        opt = self.optimizers()
        
        # first SAM pass
        enable_running_stats(self)
        y_ = self(x)
        loss_1 = self.loss_fn(y_, y)
        loss_1.backward()
        opt.first_step(zero_grad=True)
        
        # second SAM pass
        disable_running_stats(self)
        y_ = self(x)
        loss_2 = self.loss_fn(y_, y)
        loss_2.backward()
        opt.second_step(zero_grad=True)
        
        # logging
        self.log("train_loss_pre_opt", loss_1)
        self.log("train_loss_post_opt", loss_2)
        
        return loss_1
    
    def configure_optimizers(self):
        # return T.optim.SGD(self.parameters())
        return SAM(
            params = self.parameters(),
            base_optimizer = T.optim.SGD,
            lr = 0.01,
            momentum = 0.9
        )
        
    # def optimizer_step(self, *args, **kwargs):
    #     pass