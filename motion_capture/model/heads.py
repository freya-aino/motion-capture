from importlib import import_module
import torch as T
import torch.nn as nn

from motion_capture.core.torchhelpers import positional_embedding
from motion_capture.model.convolution import ConvBlock, C2f

class UpsampleCrossAttentionrNeck(nn.Module):
    def __init__(
        self, 
        output_size,
        latent_size = 1024,
        depth_multiple = 1):
        super(type(self), self).__init__()
        
        assert depth_multiple >= 1, "depth_multiple must be at least 1"
        assert latent_size / 2 > 1, "latent_size must be at least be divisible by 2"
        
        codec_latent_size = latent_size
        mid_latent_size = int(latent_size / 2)
        inner_latent_size = int(latent_size / 4)
        
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.reverse1 = nn.Sequential(
            C2f(codec_latent_size + mid_latent_size, mid_latent_size, kernel_size=1, n=int(depth_multiple), shortcut=False)
        )
        self.reverse2 = nn.Sequential(
            C2f(mid_latent_size + inner_latent_size, mid_latent_size, kernel_size=1, n=int(depth_multiple), shortcut=False),
            ConvBlock(mid_latent_size, mid_latent_size, kernel_size=3, stride=2, padding=1)
        )
        self.reverse3 = nn.Sequential(
            C2f(mid_latent_size * 2, mid_latent_size, kernel_size=1, n=int(depth_multiple), shortcut=False),
            ConvBlock(mid_latent_size, codec_latent_size, kernel_size=3, stride=2, padding=1)
        )
        
        self.Q_encoder = ConvBlock(codec_latent_size, codec_latent_size, kernel_size=1, stride=1, padding=0)
        self.K_encoder = ConvBlock(codec_latent_size, codec_latent_size, kernel_size=1, stride=1, padding=0)
        self.V_encoder = ConvBlock(codec_latent_size, codec_latent_size, kernel_size=1, stride=1, padding=0)
        # nn.Sequential(
        #     ConvBlock(codec_latent_size, codec_latent_size, kernel_size=1, stride=1, padding=0),
        #     C2f(codec_latent_size, codec_latent_size, kernel_size=1, n=int(depth_multiple), shortcut=True),
        #     ConvBlock(codec_latent_size, codec_latent_size, kernel_size=1, stride=1, padding=0),
        #     C2f(codec_latent_size, codec_latent_size, kernel_size=1, n=int(depth_multiple), shortcut=True),
        #     ConvBlock(codec_latent_size, codec_latent_size + positional_embedding_size, kernel_size=1, stride=1, padding=0)
        # )
        
        self.positional_embedding = nn.Parameter(positional_embedding(20*20, codec_latent_size), requires_grad=False)
        
        self.output_1d_conv = nn.Conv1d(codec_latent_size, output_size, kernel_size=1, stride=1, padding=0, groups=1)
        
    def forward(self, x: list):
        # x1, x2, x3 in order: middle of the backbone to final layer output
        x1, x2, x3 = x[-3:]
        
        y1 = T.cat([self.upsample_x2(x3), x2], 1)
        y1 = self.reverse1(y1)
        
        y2 = T.cat([self.upsample_x2(y1), x1], 1)
        y2 = self.reverse2(y2)
        
        y3 = T.cat([y1, y2], 1)
        y3 = self.reverse3(y3)
        
        Q = self.Q_encoder(y3).flatten(2).permute(0, 2, 1)
        Q = Q + self.positional_embedding[:Q.shape[1]].expand(Q.shape[0], -1, -1)
        
        K = self.K_encoder(y3).flatten(2).permute(0, 2, 1)
        K = K + self.positional_embedding[:K.shape[1]].expand(K.shape[0], -1, -1)
        
        V = self.V_encoder(x3).flatten(2).permute(0, 2, 1)
        
        out = nn.functional.scaled_dot_product_attention(Q, K, V)
        out = self.output_1d_conv(out.permute(0, 2, 1))
        
        return out


class AttentionHead(nn.Module):
    def __init__(self, *args, **kwargs):
        
        super(type(self), self).__init__()
        
        if type(kwargs["loss_fn"]) == str:
            self.loss_fn = getattr(T.nn.functional, kwargs["loss_fn"])
        else:
            self.loss_fn = kwargs["loss_fn"]
        
        input_shape = kwargs["input_sequence_length"], kwargs["input_dim"]
        output_shape = kwargs["output_sequence_length"], kwargs["output_dim"]
        latent_shape = kwargs["output_sequence_length"], kwargs["latent_dim"]
        
        self.input_pos_embedd = nn.Parameter(positional_embedding(*input_shape), requires_grad=False)
        self.output_pos_embedd = nn.Parameter(positional_embedding(*latent_shape), requires_grad=False)
        self.internal_state = nn.Parameter(T.randn(latent_shape, dtype=T.float32), requires_grad=True)
        
        self.encoder = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_shape[1], batch_first=True, **kwargs["transformer_encoder"]["layer"]),
                num_layers=kwargs["transformer_encoder"]["num_layers"],
            ),
            nn.Linear(input_shape[1], kwargs["latent_dim"]),
            nn.LayerNorm(kwargs["latent_dim"]),
            nn.SiLU()
        )
        
        self.decoder_1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=kwargs["latent_dim"], batch_first=True, **kwargs["transformer_decoder"]["layer"]), 
            num_layers=kwargs["transformer_decoder"]["num_layers"]
        )
        self.decoder_2 = nn.Sequential(
            nn.Linear(kwargs["latent_dim"], output_shape[1]),
            nn.LayerNorm(output_shape[1]),
            nn.SiLU()
        )
        
    def forward(self, x: T.Tensor, tgt_mask: T.Tensor = None):
        
        z = x + self.input_pos_embedd
        z = self.encoder(z)
        
        i_state = (self.internal_state + self.output_pos_embedd).expand(z.shape[0], -1, -1)
        y = self.decoder_1(i_state, z, tgt_mask=tgt_mask)
        y = self.decoder_2(y)
        
        return y
