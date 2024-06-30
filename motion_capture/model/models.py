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
        lr = 0.01,
        momentum = 0.9,
        rho = 0.5,
        base_optimizer = T.optim.SGD,
        loss_fn = T.nn.functional.l1_loss,
        depth_multiple = 1):
        
        super().__init__()
        
        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.lr = lr
        self.momentum = momentum
        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn
        self.rho = rho
        
        self.backbone = Backbone(output_channels=backbone_output_size, depth_multiple=depth_multiple)
        self.neck = UpsampleCrossAttentionrNeck(output_size=neck_output_size, latent_size=backbone_output_size, depth_multiple=depth_multiple)
        self.head = SelfAttentionHead(input_size=neck_output_size, output_size=output_size, output_length=output_length, latent_size=head_latent_size)
        
    def forward(self, x: T.Tensor):
        resnet_residuals = self.backbone(x)
        neck_out = self.neck(*resnet_residuals)
        head_out = self.head(neck_out)
        return head_out
    
    def compute_loss(self, y_, y, valid):
        loss = self.loss_fn(y_, y, reduction="none")
        loss = loss.mean(0).sum(-1) * valid
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        
        x, y, valid = batch
        opt = self.optimizers()
        
        # first SAM pass
        enable_running_stats(self)
        y_ = self(x)
        loss_1 = self.compute_loss(y_, y, valid)
        loss_1.backward()
        opt.first_step(zero_grad=True)
        
        # second SAM pass
        disable_running_stats(self)
        y_ = self(x)
        loss_2 = self.compute_loss(y_, y, valid)
        loss_2.backward()
        opt.second_step(zero_grad=True)
        
        # logging
        self.log("train_loss", loss_1)
        self.log("SAM_loss_divergence", (loss_2 - loss_1).abs())
        
        return loss_1
    
    def validation_step(self, batch, batch_idx):
        x, y, v = batch
        y_ = self(x)
        val_loss = self.compute_loss(y_, y, v)
        self.log("val_loss", val_loss.to("cpu").item(), on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        return SAM(
            params = self.parameters(),
            base_optimizer = self.base_optimizer,
            lr = self.lr,
            momentum = self.momentum,
            adaptive=True,
            rho=self.rho
        )