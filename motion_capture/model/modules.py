import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import timm
import os

from .heads import UpsampleCrossAttentionrHead


def load_timm_model(model_name: str, pretrained = True, features_only = True):
    if f"{model_name}.pth" in os.listdir("./timm_models"):
        model = T.load(f"./timm_models/{model_name}.pth").to(T.device("cpu"))
    else:
        model = timm.create_model(f"timm/{model_name}", pretrained=pretrained, features_only=features_only).to(T.device("cpu"))
        T.save(model, f"./timm_models/{model_name}.pth")
    return model

class VisionModule(pl.LightningModule):
    
    def __init__(
        self,
        
        backbone: str,
        heads: dict,
        
        # - training parameters
        optimizer_kwargs: dict = None,
        lr_scheduler_warmup_epochs: int = None,
        lr_scheduler: T.optim.lr_scheduler = None,
        lr_scheduler_kwargs: dict = None
        ):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = load_timm_model(backbone)
        self.heads = nn.ModuleDict({k: UpsampleCrossAttentionrHead(**v) for k, v in heads.items()})
        
    def forward(self, x):
        backbone_out = self.backbone(x)[-3:]
        heads_out = {}
        for k in self.heads:
            heads_out[k] = self.heads[k](backbone_out)
        return heads_out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self(x)
        total_loss = 0
        for k in self.heads:
            loss = self.heads[k].compute_loss(y_[k], y)
            self.log(f"train/{k}_loss", loss)
            total_loss += loss
        self.log("train/total_loss", total_loss)
        return total_loss
        
    def configure_optimizers(self):
        
        assert self.hparams.optimizer, "optimizer not set for training"
        
        opt = T.optim.AdamW(self.parameters(), **self.hparams.optimizer_kwargs)
        
        warmup_scheduler = T.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=self.hparams.lr_scheduler_warmup_epochs)
        scheduler = self.hparams.lr_scheduler(opt, **self.hparams.lr_scheduler_kwargs)
        lr_scheduler = T.optim.lr_scheduler.SequentialLR(opt, schedulers=[
            warmup_scheduler, 
            scheduler
        ], milestones=[self.hparams.lr_scheduler_warmup_epochs])
        
        return {
            "optimizer": opt,
            "lr_scheduler": lr_scheduler
        }