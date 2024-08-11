import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import timm
import os

from .heads import MLPHead, UpsampleCrossAttentionrHead
from .wiou.iou import IouLoss


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

def load_timm_model(model_name: str, pretrained = True, features_only = True):
    if f"{model_name}.pth" in os.listdir("./timm_models"):
        model = T.load(f"./timm_models/{model_name}.pth").to(T.device("cpu"))
    else:
        model = timm.create_model(f"timm/{model_name}", pretrained=pretrained, features_only=features_only).to(T.device("cpu"))
        T.save(model, f"./timm_models/{model_name}.pth")
    return model

class VisionModule(pl.LightningModule):
    
    def __init__(self, backbone: str, head: dict):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.backbone = load_timm_model(backbone)
        self.head = UpsampleCrossAttentionrHead(**head)
    
    def setup(self, stage: str) -> None:
        self.iouloss = IouLoss(ltype="WIoU", monotonous=False)
        
        return super().setup(stage)
    
    def forward(self, x):
        backbone_out = self.backbone(x)[-3:]
        heads_out = self.head(backbone_out)
        return heads_out
        
    def training_step(self, batch, batch_idx):
        self.iouloss.train()
        opt_head, opt_backbone = self.optimizers()
        x, y = batch
        
        # first pass
        y_ = self(x)
        
        # loss = self.head.compute_loss(y_, y)
        iloss, liou = self.iouloss(y_, y, ret_iou=True)
        
        print(iloss.shape)
        print(liou.shape)
        
        loss = iloss.mean()
        
        loss.backward()
        # opt.first_step(zero_grad=True)
        opt_head.step()
        opt_backbone.step()
        
        opt_head.zero_grad()
        opt_backbone.zero_grad()
        
        # # second pass
        # y_ = self(x)
        # loss = self.head.compute_loss(y_, y)
        # loss.backward()
        # opt.second_step(zero_grad=True)
        
        self.log("train_loss", loss)
        return loss
    
    def on_train_epoch_end(self):
        lr_scheduler_head, lr_scheduler_backbone = self.lr_schedulers()
        lr_scheduler_head.step()
        lr_scheduler_backbone.step()
        self.log("learning_rate-head", lr_scheduler_head.get_last_lr()[0])
        self.log("learning_rate-backbone", lr_scheduler_backbone.get_last_lr()[0])
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self(x)
        loss = self.head.compute_loss(y_, y)
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        # optim = SAM(
        #     self.head.parameters(), 
        #     adaptive=True, rho=0.4, base_optimizer=T.optim.SGD, 
        #     lr=0.1, momentum=0.9, weight_decay=0.0005)
        optim_head = T.optim.AdamW(self.head.parameters(), lr=1e-2)
        optim_backbone = T.optim.AdamW(self.backbone.parameters(), lr=1e-2)
        
        lr_scheduler_head = T.optim.lr_scheduler.CosineAnnealingLR(optim_head, T_max=50, eta_min=1e-4)
        lr_scheduler_backbone = T.optim.lr_scheduler.CosineAnnealingLR(optim_backbone, T_max=50, eta_min=1e-6)
        
        return [optim_head, optim_backbone], [lr_scheduler_head, lr_scheduler_backbone]
        
        # warmup_scheduler = T.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=self.hparams.lr_scheduler_warmup_epochs)
        # scheduler = T.optim.lr_scheduler.(opt, **self.hparams.lr_scheduler_kwargs)
        # lr_scheduler = T.optim.lr_scheduler.SequentialLR(opt, schedulers=[
        #     warmup_scheduler, 
        #     scheduler
        # ], milestones=[self.hparams.lr_scheduler_warmup_epochs])