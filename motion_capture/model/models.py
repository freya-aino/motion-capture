import os
import time
import torch as T
import torch.nn as nn
import pytorch_lightning as pl

from torchvision import models as torchModels

from .heads import AttentionHead


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

class MainModel(pl.LightningModule):

    def __init__(
        self,
        
        heads: dict,
        
        # - training parameters
        train_backbone: bool = None,
        optimizer: T.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        lr_scheduler_warmup_epochs: int = None,
        lr_scheduler: T.optim.lr_scheduler = None,
        lr_scheduler_kwargs: dict = None,
        loss_fn = None,
        invalid_element_loss_scale = None):
        
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        
        self.backbone = torchModels.convnext_tiny(weights=torchModels.ConvNeXt_Tiny_Weights.DEFAULT).features
        self.heads = nn.ModuleDict({
            k: AttentionHead(
                input_size = 768, # base input size of convnext_tiny and convnext_small
                output_size = heads[k]["output_size"], 
                output_length= heads[k]["output_length"],
                latent_size= heads[k]["latent_size"],
                depth_multiple= heads[k]["depth_multiple"],
            ) for k in heads
        })
    
    def forward(self, x: T.Tensor):
        return self.head(self.backbone(x))
    
    def compute_loss(self, y_, y, loss_fn):
        return loss_fn(y_, y)
    
    def head_wise_step(self, batch, mode):
        
        if self.hparams.train_backbone:
            self.backbone.eval()
        
        x, y = batch
        outputs = self(x)
        losses = {}
        for k, head_output in outputs.items():
            loss = self.compute_loss(head_output, y[k], self.hparams.heads[k]["loss_fn"])
            losses[k] = loss
            self.log(f"{mode}_loss_{k}", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f"{mode}_loss", sum(losses.values()) / len(losses), on_step=False, on_epoch=True, prog_bar=True)
        return losses
    
    def training_step(self, batch, batch_id):
        return self.head_wise_step(batch, "train")
    def validation_step(self, batch, batch_idx):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=False)
        return self.head_wise_step(batch, "val")
    def test_step(self, batch, batch_idx):
        return self.head_wise_step(batch, "test")
    
    def configure_optimizers(self):
        
        assert self.hparams.optimizer, "optimizer not set for training"
        
        opt = self.hparams.optimizer(self.parameters(), **self.hparams.optimizer_kwargs)
        
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