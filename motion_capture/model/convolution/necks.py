import math
import torch as T
import torch.nn as nn

from .core import ConvBlock, C2f
from motion_capture.core.torchhelpers import positional_embedding


class PyramidTransformerNeck(nn.Module):
    '''
        based on this https://arxiv.org/pdf/2207.03917.pdf
        
        TODO: not finished yet, but it has no priority rn (28.10.2023)
        
    '''
    def __init__(
        self, 
        output_size,
        depth_multiple = 0.33,
        width = 1024,
        width_multiple = 0.25,
        original_positional_embedding_size: int = 256):
        
        super(type(self), self).__init__()
        
        raise NotImplementedError
        
        self.latent_sizes = {
            "encoder": int(width * width_multiple),
            "hidden": int(512 * width_multiple),
            "hidden_3": int(128 * width_multiple),
            "hidden_2": int(256 * width_multiple),
        }
        
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.POS_ENC3 = positional_embedding(20*20, self.latent_sizes["encoder"])
        
        self.POS_ENC2 = positional_embedding(40*40, self.latent_sizes["hidden"])
        self.MHA2 = nn.MultiheadAttention(embed_dim=self.latent_sizes["hidden"], num_heads=8)
        self.LN2_1 = nn.LayerNorm(self.latent_sizes["hidden"])
        self.FFN2 = nn.Linear(self.latent_sizes["hidden"], self.latent_sizes["hidden"])
        self.LN2_2 = nn.LayerNorm(self.latent_sizes["hidden"])
        
        self.POS_ENC1 = positional_embedding(80*80, self.latent_sizes["hidden_2"])
        self.MHA1 = nn.MultiheadAttention(embed_dim=self.latent_sizes["hidden_2"], num_heads=8)
        self.LN1_1 = nn.LayerNorm(self.latent_sizes["hidden_2"])
        self.FFN1 = nn.Linear(self.latent_sizes["hidden_2"], self.latent_sizes["hidden_3"])
        self.LN1_2 = nn.LayerNorm(self.latent_sizes["hidden_3"])
        
        self.LL_MSA3 = nn.MultiheadAttention(self.latent_sizes["encoder"], num_heads=8)
        self.LM_MSA3 = nn.MultiheadAttention(self.latent_sizes["encoder"], num_heads=8)
        
        self.LL_MSA2 = nn.MultiheadAttention(self.latent_sizes["hidden"], num_heads=8)
        self.LM_MSA2 = nn.MultiheadAttention(self.latent_sizes["hidden"], num_heads=8)
        
        self.LL_MSA1 = nn.MultiheadAttention(self.latent_sizes["hidden_2"], num_heads=8)
        self.LM_MSA1 = nn.MultiheadAttention(self.latent_sizes["hidden_2"], num_heads=8)
        
    def forward(self, x1: T.Tensor, x2: T.Tensor, x3: T.Tensor) -> T.Tensor: 
        
        # cross scale attention 3
        v3 = x3
        
        # cross scale attention 2
        value2 = (x2 * self.upsample_x2(v3)).flatten(2).permute(0, 2, 1)
        key2 = (x2 * self.upsample_x2(v3)).flatten(2).permute(0, 2, 1)
        key2 = key2 + self.POS_ENC2[:key2.shape[1]].expand(key2.shape[0], -1, -1)
        query2 = x2.flatten(2).permute(0, 2, 1) 
        query2 = query2 + self.POS_ENC2[:query2.shape[1]].expand(query2.shape[0], -1, -1)
        
        v2 = self.LN2_1(query2 + self.MHA2(query2, key2, value2))
        v2 = self.LN2_2(v2 + self.FFN2(v2))
        v2 = v2.reshape(v2.shape[0], v2.shape[1], int(math.sqrt(v2.shape[2])), int(math.sqrt(v2.shape[2])))
        
        # cross scale attention 1
        value1 = (x1 * self.upsample_x2(v2)).flatten(2).permute(0, 2, 1)
        key1 = (x1 * self.upsample_x2(v2)).flatten(2).permute(0, 2, 1)
        key1 = key1 + self.POS_ENC1[:key1.shape[1]].expand(key1.shape[0], -1, -1)
        query1 = x1.flatten(2).permute(0, 2, 1)
        query1 = query1 + self.POS_ENC1[:query1.shape[1]].expand(query1.shape[0], -1, -1)
        
        v1 = self.LN1_1(query1 + self.MHA1(query1, key1, value1))
        v1 = self.LN1_2(v1 + self.FFN1(v1))
        v1 = v1.reshape(v1.shape[0], v1.shape[1], int(math.sqrt(v1.shape[2])), int(math.sqrt(v1.shape[2])))
        
        # pyramid transformer head for v3
        qk_ll3 = x3.flatten(2)
        qk_ll3 = qk_ll3 + self.POS_ENC3[:qk_ll3.shape[1]].expand(qk_ll3.shape[0], -1, -1)
        v_ll3 = x3.flatten(2)
        LL3 = self.LL_MSA3(qk_ll3, qk_ll3, v_ll3)
        
        q_ml3 = LL3
        q_ml3 = q_ml3 + self.POS_ENC3[:q_ml3.shape[1]].expand(q_ml3.shape[0], -1, -1)
        k_ml3 = v3.flatten(2)
        k_ml3 = k_ml3 + self.POS_ENC3[:k_ml3.shape[1]].expand(k_ml3.shape[0], -1, -1)
        v_ml3 = v3.flatten(2)
        LM3 = self.LM_MSA3(q_ml3, k_ml3, v_ml3)
        
        # pyramid transformer head for v2
        qk_ll2 = x2.flatten(2)
        qk_ll2 = qk_ll2 + self.POS_ENC2[:qk_ll2.shape[1]].expand(qk_ll2.shape[0], -1, -1)
        v_ll2 = x2.flatten(2)
        LL2 = self.LL_MSA2(qk_ll2, qk_ll2, v_ll2)
        
        q_ml2 = LL2
        q_ml2 = q_ml2 + self.POS_ENC2[:q_ml2.shape[1]].expand(q_ml2.shape[0], -1, -1)
        k_ml2 = v2.flatten(2)
        k_ml2 = k_ml2 + self.POS_ENC2[:k_ml2.shape[1]].expand(k_ml2.shape[0], -1, -1)
        v_ml2 = v2.flatten(2)
        LM2 = self.LM_MSA2(q_ml2, k_ml2, v_ml2)
        
        # pyramid transformer head for v1
        
        # TODO put all these components in in dividual modules, this is horrible repetition
        
        # TODO add dynamic landmark refinement for model output
        
        pass


class UpsampleCrossAttentionrNeck(nn.Module):
    def __init__(
        self, 
        output_size,
        depth_multiple = 0.33,
        width = 1024,
        width_multiple = 0.25,
        original_positional_embedding_size: int = 256):
        super(type(self), self).__init__()
        
        self.latent_sizes = {
            "codec": int(width * width_multiple), # both for encoder and decoder input/output sizes
            "latent": int(width / 2 * width_multiple), # for the latent space of the encoder/decoder
            "latent_2": int(width / 4 * width_multiple), # latent space
        }
        scaled_positional_embedding_size = int(original_positional_embedding_size * width_multiple)
        
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.reverse1 = nn.Sequential(
            C2f(self.latent_sizes["codec"] + self.latent_sizes["latent"], self.latent_sizes["latent"], kernel_size=1, n=int(3*depth_multiple), shortcut=False)
        )
        self.reverse2 = nn.Sequential(
            C2f(self.latent_sizes["latent"] + self.latent_sizes["latent_2"], self.latent_sizes["latent"], kernel_size=1, n=int(3*depth_multiple), shortcut=False),
            ConvBlock(self.latent_sizes["latent"], self.latent_sizes["latent"], kernel_size=3, stride=2, padding=1)
        )
        self.reverse3 = nn.Sequential(
            C2f(self.latent_sizes["latent"] * 2, self.latent_sizes["latent"], kernel_size=1, n=int(3*depth_multiple), shortcut=False),
            ConvBlock(self.latent_sizes["latent"], self.latent_sizes["codec"], kernel_size=3, stride=2, padding=1)
        )
        
        self.Q_encoder = ConvBlock(self.latent_sizes["codec"], self.latent_sizes["codec"], kernel_size=1, stride=1, padding=0)
        self.K_encoder = ConvBlock(self.latent_sizes["codec"], self.latent_sizes["codec"], kernel_size=1, stride=1, padding=0)
        self.V_encoder = nn.Sequential(
            # ConvBlock(self.latent_sizes["codec"], self.latent_sizes["codec"], kernel_size=1, stride=1, padding=0),
            # C2f(self.latent_sizes["codec"], self.latent_sizes["codec"], kernel_size=1, n=int(3*depth_multiple), shortcut=True),
            # ConvBlock(self.latent_sizes["codec"], self.latent_sizes["codec"], kernel_size=1, stride=1, padding=0),
            # C2f(self.latent_sizes["codec"], self.latent_sizes["codec"], kernel_size=1, n=int(3*depth_multiple), shortcut=True),
            ConvBlock(self.latent_sizes["codec"], self.latent_sizes["codec"] + scaled_positional_embedding_size, kernel_size=1, stride=1, padding=0)
        )
        
        self.positional_embedding = nn.Parameter(positional_embedding(20*20, scaled_positional_embedding_size), requires_grad=False)
        
        self.output_1d_conv = nn.Conv1d(self.latent_sizes["codec"] + scaled_positional_embedding_size, output_size, kernel_size=1, stride=1, padding=0, groups=1)
        
    def forward(self, x1: T.Tensor, x2: T.Tensor, x3: T.Tensor) -> T.Tensor: 
        # x1, x2, x3 in order: middle of the backbone to final layer output
        
        y1 = T.cat([self.upsample_x2(x3), x2], 1)
        y1 = self.reverse1(y1)
        
        y2 = T.cat([self.upsample_x2(y1), x1], 1)
        y2 = self.reverse2(y2)
        
        y3 = T.cat([y1, y2], 1)
        y3 = self.reverse3(y3)
        
        Q = self.Q_encoder(y3).flatten(2).permute(0, 2, 1)
        Q = T.cat([Q, self.positional_embedding[:Q.shape[1]].expand(Q.shape[0], -1, -1)], 2)
        
        K = self.K_encoder(y3).flatten(2).permute(0, 2, 1)
        K = T.cat([K, self.positional_embedding[:K.shape[1]].expand(K.shape[0], -1, -1)], 2)
        
        V = self.V_encoder(x3).flatten(2).permute(0, 2, 1)
        
        out = nn.functional.scaled_dot_product_attention(Q, K, V)
        out = self.output_1d_conv(out.permute(0, 2, 1))
        
        return out
        