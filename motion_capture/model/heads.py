from importlib import import_module
import torch as T
import torch.nn as nn

from motion_capture.core.torchhelpers import positionalencoding1d
from motion_capture.model.convolution import ConvBlock, C2f


class VQVAEHead(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__()
        
        input_dim = kwargs["input_dim"]
        output_dim = kwargs["output_dim"]
        output_sequence_length = kwargs["output_sequence_length"]
        
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

class UpsampleCrossAttentionrHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(type(self), self).__init__()
        
        input_dim = kwargs["input_dim"]
        depth = kwargs["depth"]
        output_dim = kwargs["output_dim"]
        input_sequence_length = kwargs["input_sequence_length"]
        
        self.output_sequence_length = kwargs["output_sequence_length"]
        self.continuous_output = kwargs["continuous_output"]
        
        assert depth >= 1, "depth must be at least 1"
        
        l0 = input_dim
        l1 = input_dim // 2
        l2 = input_dim // 4
        
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.reverse1 = C2f(l0 + l1, l1, kernel_size=1, n=depth, shortcut=True)
        self.reverse2 = nn.Sequential(
            C2f(l1 + l2, l2, kernel_size=1, n=depth, shortcut=True),
            ConvBlock(l2, l2, kernel_size=3, stride=2, padding=1)
        )
        self.reverse3 = nn.Sequential(
            C2f(l1 + l2, l0, kernel_size=1, n=depth, shortcut=True),
            ConvBlock(l0, l0, kernel_size=3, stride=2, padding=1)
        )
        
        self.att = nn.MultiheadAttention(l0, 1, batch_first=True, dropout=0)
        self.internal_state = nn.Parameter(positionalencoding1d(input_dim, self.output_sequence_length), requires_grad=False)
        
        shape_progression = [int(l0 + (output_dim - l0) * (i / depth))  for i in range(depth + 1)]
        self.decoder = nn.Sequential(*[
            nn.Sequential(nn.Linear(d1, d2), nn.LayerNorm(d2), nn.SiLU()) 
            for (d1, d2) in zip(shape_progression[:-1], shape_progression[1:])]
        )
        
    def forward(self, x: list):
        x1, x2, x3 = x[-3:]
        
        y1 = T.cat([self.upsample_x2(x3), x2], 1)
        y1 = self.reverse1(y1)
        
        y2 = T.cat([self.upsample_x2(y1), x1], 1)
        y2 = self.reverse2(y2)
        
        y3 = T.cat([y1, y2], 1)
        y3 = self.reverse3(y3)
        
        internal_state = self.internal_state.expand(y3.shape[0], -1, -1)
        
        kv = T.cat([y3.flatten(2).permute(0, 2, 1), internal_state], 1)
        q = x3.flatten(2).permute(0, 2, 1)
        
        out, _ = self.att(query=q, key=kv, value=kv)
        out = self.decoder(out[:, -self.output_sequence_length:, :])
        return out
    
    def compute_loss(self, y_pred, y):
        loss_fn = T.nn.functional.smooth_l1_loss if self.continuous_output else T.nn.functional.cross_entropy
        return loss_fn(y_pred, y)

# class AttentionHead(nn.Module):
#     def __init__(self, *args, **kwargs):
        
#         super(type(self), self).__init__()
        
#         if type(kwargs["loss_fn"]) == str:
#             self.loss_fn = getattr(T.nn.functional, kwargs["loss_fn"])
#         else:
#             self.loss_fn = kwargs["loss_fn"]
        
#         input_shape = kwargs["input_sequence_length"], kwargs["input_dim"]
#         output_shape = kwargs["output_sequence_length"], kwargs["output_dim"]
#         latent_shape = kwargs["output_sequence_length"], kwargs["latent_dim"]
        
#         self.input_pos_embedd = nn.Parameter(positional_embedding(*input_shape), requires_grad=False)
#         self.output_pos_embedd = nn.Parameter(positional_embedding(*latent_shape), requires_grad=False)
#         self.internal_state = nn.Parameter(T.randn(latent_shape, dtype=T.float32), requires_grad=True)
        
#         self.encoder = nn.Sequential(
#             nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(d_model=input_shape[1], batch_first=True, **kwargs["transformer_encoder"]["layer"]),
#                 num_layers=kwargs["transformer_encoder"]["num_layers"],
#             ),
#             nn.Linear(input_shape[1], kwargs["latent_dim"]),
#             nn.LayerNorm(kwargs["latent_dim"]),
#             nn.SiLU()
#         )
        
#         self.decoder_1 = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=kwargs["latent_dim"], batch_first=True, **kwargs["transformer_decoder"]["layer"]), 
#             num_layers=kwargs["transformer_decoder"]["num_layers"]
#         )
#         self.decoder_2 = nn.Sequential(
#             nn.Linear(kwargs["latent_dim"], output_shape[1]),
#             nn.LayerNorm(output_shape[1]),
#             nn.SiLU()
#         )
        
#     def forward(self, x: T.Tensor, tgt_mask: T.Tensor = None):
        
#         z = x + self.input_pos_embedd
#         z = self.encoder(z)
        
#         i_state = (self.internal_state + self.output_pos_embedd).expand(z.shape[0], -1, -1)
#         y = self.decoder_1(i_state, z, tgt_mask=tgt_mask)
#         y = self.decoder_2(y)
        
#         return y
