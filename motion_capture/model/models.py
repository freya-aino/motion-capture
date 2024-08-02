import os
import time
import torch as T
import torch.nn as nn
import pytorch_lightning as pl
import timm

from ..core.torchhelpers import positional_embedding
from .convolution import ConvBlock, C2f, SPPF


class VisionModel(pl.LightningModule):
    
    def __init__(
        self,
        
        backbone: str,
        vqvae: dict,
        heads: dict,
        
        # - training parameters
        optimizer: T.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        lr_scheduler_warmup_epochs: int = None,
        lr_scheduler: T.optim.lr_scheduler = None,
        lr_scheduler_kwargs: dict = None,
        
        # - loss scale parameters
        reconstruction_loss_scale: float = 1.0,
        codebook_loss_scale: float = 1.0,
        prediction_loss_scales: dict = None
        ):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        self.vqvae = vqvae = VQVAE(**vqvae)
        
        self.heads = nn.ModuleDict({k: VQVAEHead(**v) for k, v in heads.items()})
        
    def forward(self, x, skip_backbone=False):
        
        if not skip_backbone:
            backbone_out = self.backbone(x)[-1].flatten(2).permute(0, 2, 1)
        else:
            backbone_out = x
        
        vqvae_out = self.vqvae(backbone_out)
        
        heads_out = {}
        for k in self.heads:
            heads_out[k] = self.heads[k](vqvae_out["codebook_onehots"])
        
        return {
            "heads": heads_out, 
            "vqvae": vqvae_out
        }
    
    def get_losses(self, 
        heads_out, heads_targets,
        vqvae_out, vqvae_target):
        
        reconstruction_loss, codebook_loss = self.vqvae.compute_loss(vqvae_reconstruction_target, vqvae_out["reconstruction"], vqvae_out["z"], vqvae_out["codebook_indecies"])
        prediction_losses = {
            k: self.hparams.prediction_loss_scales.get(k, 1) * self.heads[k].compute_loss(heads_targets[k], heads_out[k]) 
            for k in self.heads
        }
        
        return {
            "vqvae-reconstruction": reconstruction_loss * self.hparams.reconstruction_loss_scale,
            "vqvae-codebook": codebook_loss * self.hparams.codebook_loss_scale,
            **prediction_losses
        }


# class UpsampleCrossAttentionrHead(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(type(self), self).__init__()
        
#         input_dim = kwargs["input_dim"]
#         depth = kwargs["depth"]
        
#         assert depth >= 1, "depth must be at least 1"
        
#         l0 = input_dim
#         l1 = input_dim // 2
#         l2 = input_dim // 4
        
#         self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
#         self.reverse1 = C2f(l0 + l1, l1, kernel_size=1, n=depth, shortcut=True)
#         self.reverse2 = nn.Sequential(
#             C2f(l1 + l2, l2, kernel_size=1, n=depth, shortcut=True),
#             ConvBlock(l2, l2, kernel_size=3, stride=2, padding=1)
#         )
#         self.reverse3 = nn.Sequential(
#             C2f(l1 + l2, l0, kernel_size=1, n=depth, shortcut=True),
#             ConvBlock(l0, l0, kernel_size=3, stride=2, padding=1)
#         )
        
#         self.tf = nn.Transformer(**kwargs["transformer"])
        
#     def forward(self, x: list):
#         # x1, x2, x3 in order: middle of the backbone to final layer output
#         x1, x2, x3 = x[-3:]
        
#         # x3 = self.sppf(x3)
        
#         y1 = T.cat([self.upsample_x2(x3), x2], 1)
#         y1 = self.reverse1(y1)
        
#         y2 = T.cat([self.upsample_x2(y1), x1], 1)
#         y2 = self.reverse2(y2)
        
#         y3 = T.cat([y1, y2], 1)
#         y3 = self.reverse3(y3)
        
#         kv = y3.flatten(2).permute(0, 2, 1)
#         q = x3.flatten(2).permute(0, 2, 1)
#         # kv = self.kv_encode(y3.flatten(2)).permute(0, 2, 1)
#         # q = self.q_encode(x3.flatten(2)).permute(0, 2, 1)
        
#         # kv = kv + self.positional_embedding.expand(kv.size(0), -1, -1)
#         # q = q + self.positional_embedding.expand(q.size(0), -1, -1)
        
#         out, _ = self.att(query=q, key=kv, value=kv, attn_mask = self.mask)
#         # out = self.output_encoder(out)
        
#         return out


class VQVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__()
        
        input_dim = kwargs["input_dim"]
        codebook_dim = kwargs["codebook_dim"]
        num_codebook_entries = kwargs["num_codebook_entries"]
        output_dim = kwargs["output_dim"]
        
        codebook_sequence_length = kwargs["codebook_sequence_length"]
        output_sequence_length = kwargs["output_sequence_length"]
        
        self.input_state = nn.Parameter(T.randn(codebook_sequence_length, codebook_dim, dtype=T.float32), requires_grad=True)
        self.codebook = nn.Embedding(num_codebook_entries, codebook_dim)
        self.output_state = nn.Parameter(T.randn(output_sequence_length, codebook_dim, dtype=T.float32), requires_grad=True)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim + codebook_dim) // 2),
            nn.LayerNorm((input_dim + codebook_dim) // 2),
            nn.SiLU(),
            nn.Linear((input_dim + codebook_dim) // 2, codebook_dim),
            nn.LayerNorm(codebook_dim),
            nn.SiLU(),
        )
        
        self.tf_encoder = nn.Transformer(d_model = codebook_dim, **kwargs["transformer"], batch_first = True)
        self.tf_decoder = nn.Transformer(d_model = codebook_dim, **kwargs["transformer"], batch_first = True)
        
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, (output_dim + codebook_dim) // 2),
            nn.LayerNorm((output_dim + codebook_dim) // 2),
            nn.SiLU(),
            nn.Linear((output_dim + codebook_dim) // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        
    def forward(self, x: T.Tensor):
        
        x = self.encoder(x)
        
        z = self.tf_encoder(src = x, tgt = self.input_state.expand(x.shape[0], -1, -1))
        
        c = T.cdist(z, self.codebook.weight, p = 2).argmin(-1)
        z_ = self.codebook(c)
        
        rec = None
        if self.train:
            rec = self.tf_decoder(src = z_, tgt = self.output_state.expand(z.shape[0], -1, -1))
            rec = self.decoder(rec)
        
        return {
            "z": z,
            "discrete_z": z_,
            "codebook_indecies": c,
            "codebook_onehots": nn.functional.one_hot(c, self.codebook.num_embeddings).to(dtype=T.float32),
            "reconstruction": rec
        }
    
    def compute_loss(self, y, y_pred, z, codebook_indecies):
        loss_fn = T.nn.functional.l1_loss
        return loss_fn(y_pred, y), loss_fn(z, self.codebook(codebook_indecies))


class VQVAEHead(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__()
        
        input_dim = kwargs["input_dim"]
        output_dim = kwargs["output_dim"]
        output_sequence_length = kwargs["output_sequence_length"]
        
        self.continuous_output = kwargs["continuous_output"]
        
        self.target_sequence = nn.Parameter(T.randn(output_sequence_length, input_dim, dtype=T.float32), requires_grad=True)
        
        self.tf = nn.Transformer(d_model = input_dim, **kwargs["transformer"], batch_first = True)
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim + output_dim) // 2), 
            nn.LayerNorm((input_dim + output_dim) // 2),
            nn.SiLU(),
            nn.Linear((input_dim + output_dim) // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU() if self.continuous_output else nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        z = self.tf(src = x, tgt = self.target_sequence.expand(x.shape[0], -1, -1))
        out = self.decoder(z)
        return out
    
    def compute_loss(self, y_pred, y):
        loss_fn = T.nn.functional.l1_loss if self.continuous_output else T.nn.functional.cross_entropy
        return loss_fn(y_pred, y)

