import os
import time
import torch as T
import torch.nn as nn
import pytorch_lightning as pl

from .convolution.backbones import Backbone
from .convolution.necks import UpsampleCrossAttentionrNeck
from .convolution.heads import SelfAttentionHead
from .SAM.sam import SAM
from .SAM.example.utility.bypass_bn import enable_running_stats, disable_running_stats


def find_best_checkpoint_path(checkpoint_dir, min_loss: bool = True, pattern="*.ckpt"):
    from glob import glob
    import re
    
    files = glob(os.path.join(checkpoint_dir, pattern))
    
    if len(files) == 0:
        return None
    
    all_models = []
    for file in files:
        ckpt = T.load(file, map_location=T.device("cpu"))
        for key, val in ckpt.get("callbacks", {}).items():
            if key.startswith("ModelCheckpoint"):
                all_models.append({
                    "model_path": val["best_model_path"],
                    "model_score": val["best_model_score"]
                })
    if min_loss:
        best_model = min(all_models, key=lambda x: x["model_score"])
    else:
        best_model = max(all_models, key=lambda x: x["model_score"])
    
    print(f"found best model with loss: {best_model['model_score']} from {best_model['model_path']}")
    return best_model["model_path"]

class UpsampleCrossAttentionNetwork(pl.LightningModule):

    def __init__(
        self,
        
        # - model parameters
        output_size: int,
        output_length: int,
        backbone_output_size: int,
        neck_output_size: int,
        head_latent_size: int,
        depth_multiple: int,
        
        # - training parameters
        optimizer: T.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        lr_scheduler: T.optim.lr_scheduler = None,
        lr_scheduler_kwargs: dict = None,
        loss_fn = None,
        ):
        
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        
        self.backbone = Backbone(output_channels=backbone_output_size, depth_multiple=depth_multiple)
        self.neck = UpsampleCrossAttentionrNeck(output_size=neck_output_size, latent_size=backbone_output_size, depth_multiple=depth_multiple)
        self.head = SelfAttentionHead(input_size=neck_output_size, output_size=output_size, output_length=output_length, latent_size=head_latent_size)
    
    def new_head(self, new_output_size, new_output_length):
        self.head = SelfAttentionHead(input_size=self.hparams.neck_output_size, output_size=new_output_size, output_length=new_output_length, latent_size=self.hparams.head_latent_size)
    def replace_head(self, new_head: nn.Module):
        self.head = new_head
        
    def new_neck(self, new_neck_output_size):
        self.neck = UpsampleCrossAttentionrNeck(output_size=new_neck_output_size, latent_size=self.hparams.backbone_output_size, depth_multiple=self.hparams.depth_multiple)
    def replace_neck(self, new_neck: nn.Module):
        self.neck = new_neck
        
    def forward(self, x: T.Tensor):
        resnet_residuals = self.backbone(x)
        neck_out = self.neck(*resnet_residuals)
        head_out = self.head(neck_out)
        return head_out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        train_loss = self.hparams.loss_fn(self(x), y)
        self.log("train_loss", train_loss, on_step=True, on_epoch=False, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.hparams.loss_fn(self(x), y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_param_variances()
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        test_loss = self.hparams.loss_fn(self(x), y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss
    
    def configure_optimizers(self):
        
        assert self.hparams.optimizer, "optimizer not set for training"
        
        opt = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_kwargs)
        lr_scheduler = self.hparams.lr_scheduler(opt, **self.hparams.lr_scheduler_kwargs)
        
        return {
            "optimizer": opt,
            "lr_scheduler": lr_scheduler
        }
    
    def log_param_variances(self):
        for name, param in self.named_parameters():
            if "weight" in name:  # Filter to log variance of weights only
                variance = T.var(param).item()
                self.log(f"{name}_variance", variance, on_epoch=True, on_step=False, prog_bar=False)