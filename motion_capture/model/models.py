import os
import time
import torch as T
import torch.nn as nn


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
