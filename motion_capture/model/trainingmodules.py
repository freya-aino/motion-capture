import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import timm
import os
import torch.nn.functional as F

from .wiou.iou import IouLoss
from .SAM.sam import SAM
from motion_capture.model.heads import PyramidTransformerHead
from motion_capture.core.utils import load_timm_model

class BBoxTrainingModule(pl.LightningModule):
    
    def __init__(
        self, 
        backbone_name: str, 
        head_kwargs: dict, 
        iou_loss_type="IoU", 
        finetune=False,
        optimizer_kwargs: dict = {},
        lr_scheduler_kwargs: dict = {}
        ):
        super().__init__()
        
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.backbone = load_timm_model(backbone_name, pretrained=True, features_only=True)
        self.head = PyramidTransformerHead(**head_kwargs)
        self.iouloss = IouLoss(ltype=iou_loss_type, monotonous=None)
        self.finetune = finetune
        
        if not finetune:
            self.backbone.eval()
    
    def forward(self, x):
        backbone_out = self.backbone(x)[-3:]
        return self.head(backbone_out)
    
    def compute_loss(self, y_, y):
        
        y = y.reshape(-1, 4)
        
        area = (y[..., 2:4] - y[..., 0:2]).prod(dim=-1)
        target_area_valid = ~T.isclose(area, T.tensor(0.0))
        
        loss, iou = 0, 0
        valids = 0
        for y_i in y_:
            
            y_i = y_i.reshape(-1, 4)
            
            area = (y_i[..., 2:4] - y_i[..., 0:2]).prod(dim=-1)
            
            predicted_area_valid = ~T.isclose(area, T.tensor(0.0))
            m = predicted_area_valid & target_area_valid
            
            if m.sum() == 0:
                continue
            valids += 1
            
            # print("y_i[m]: ", y_i[m].shape)
            # print("y[m]: ", y[m].shape)
            
            losses = self.head.compute_loss(y_i[m], y[m], loss_fn=self.iouloss, ret_iou=True)
            
            ldif, liou = [*zip(*losses)]
            ldif = T.stack(ldif)
            liou = T.stack(liou)
            
            # print("ldif: ", ldif.shape)
            # print("liou: ", liou.shape)
            
            loss += (ldif - 1).mean()
            iou += (1 - liou).mean()
        
        loss /= valids
        iou /= valids
        
        return loss, iou
        
        # return self.head.calculate_loss(y_[m], y[m], loss_fn=F.l1_loss)
    
    def on_train_start(self):
        if not self.finetune:
            self.backbone.eval()
        else:
            self.backbone.train()
        
        self.head.train()
        self.iouloss.train()
        
        return super().on_train_start()
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        
        # --- SAM ---
        def closure():
            loss, iou = self.compute_loss(self(x), y)
            if loss == 0:
                print("no valid area")
                return None
            loss.backward()
            return loss
        
        loss, iou = self.compute_loss(self(x), y)
        if loss == 0:
            print("no valid area")
            return None
        loss.backward()
        opt.step(closure)
        opt.zero_grad()
        
        
        self.log("train_loss", loss)
        self.log("train_IoU", iou)
        
        return {
            "train_loss": loss,
            "train_IoU": iou
        }
    
    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.log("learning_rate", lr_scheduler.get_last_lr()[0])
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        loss, iou = self.compute_loss(self(x), y)
        if loss == 0:
            print("no valid area")
            return None
        
        self.log("val_loss", loss)
        self.log("val_IoU", iou)
        return {
            "val_loss": loss
        }
    
    def configure_optimizers(self):
        if self.finetune:
            optim = SAM(self.parameters(), base_optimizer=T.optim.SGD, **self.hparams.optimizer_kwargs)
        else:
            optim = SAM(self.head.parameters(), base_optimizer=T.optim.SGD, **self.hparams.optimizer_kwargs)
        
        lr_scheduler = T.optim.lr_scheduler.CosineAnnealingLR(optim, **self.hparams.lr_scheduler_kwargs)
        
        return [optim], [lr_scheduler]