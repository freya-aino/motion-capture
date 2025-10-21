import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Args, Kwargs

from motion_capture.components.convolution import ConvBlock, C2f, Detection, SPPF
from motion_capture.components.transformer import PyramidTransformer, LL_LM_Attention


class Backbone(nn.Module):
    def __init__(self, output_channels: int, depth: int = 1, input_channels: int = 3):
        super(type(self), self).__init__()

        assert output_channels / 16 >= 1, (
            "output_channels must be at least be divisible by 16"
        )
        assert depth >= 1, "depth_multiple must be at least 1"

        size_4 = output_channels
        size_3 = output_channels // 2
        size_2 = output_channels // 4
        size_1 = output_channels // 8
        size_0 = output_channels // 16

        self.conv1 = nn.Sequential(
            ConvBlock(input_channels, size_0, kernel_size=3, stride=2, padding=1),
            ConvBlock(size_0, size_1, kernel_size=3, stride=2, padding=1),
            C2f(size_1, size_1, kernel_size=1, n=depth, shortcut=True),
            ConvBlock(size_1, size_2, kernel_size=3, stride=2, padding=1),
            C2f(size_2, size_2, kernel_size=1, n=depth, shortcut=True),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(size_2, size_3, kernel_size=3, stride=2, padding=1),
            C2f(size_3, size_3, kernel_size=1, n=2 * depth, shortcut=True),
        )

        self.conv3 = nn.Sequential(
            ConvBlock(size_3, size_4, kernel_size=3, stride=2, padding=1),
            C2f(size_4, size_4, kernel_size=1, n=depth, shortcut=True),
            SPPF(size_4, size_4),
        )

        self.batch_norm_x1 = nn.BatchNorm2d(size_2)
        self.batch_norm_x2 = nn.BatchNorm2d(size_3)
        self.batch_norm_x3 = nn.BatchNorm2d(size_4)

    def forward(self, x: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x1 = self.batch_norm_x1(x1)
        x2 = self.batch_norm_x2(x2)
        x3 = self.batch_norm_x3(x3)

        return x1, x2, x3


class YOLOv8Head(nn.Module):
    # model head from: https://blog.roboflow.com/whats-new-in-yolov8/

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        output_lenght: int,
        num_classes: int,
        depth: int = 1,
    ):
        super(type(self), self).__init__()

        assert depth >= 1, "depth must be at least 1"

        l0 = input_channels
        l1 = input_channels // 2
        l2 = input_channels // 4

        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.c2f_1 = C2f(l0 + l1, l1, kernel_size=1, n=depth, shortcut=False)
        self.c2f_2 = C2f(l1 + l2, l2, kernel_size=1, n=depth, shortcut=False)

        self.conv_1 = ConvBlock(l2, l2, kernel_size=3, stride=2, padding=1)
        self.c2f_3 = C2f(l2 + l1, l1, kernel_size=1, n=depth, shortcut=False)

        self.conv_2 = ConvBlock(l1, l1, kernel_size=3, stride=2, padding=1)
        self.c2f_4 = C2f(l1 + l0, l1, kernel_size=1, n=depth, shortcut=False)

        self.det_1 = Detection(l2, output_lenght, num_classes)
        self.det_2 = Detection(l1, output_lenght, num_classes)
        self.det_3 = Detection(l1, output_lenght, num_classes)

    def forward(self, x: list):
        x1, x2, x3 = x

        z = T.cat([self.upsample_x2(x3), x2], 1)
        z = self.c2f_1(z)

        y1 = T.cat([self.upsample_x2(z), x1], 1)
        y1 = self.c2f_2(y1)

        y2 = T.cat([self.conv_1(y1), z], 1)
        y2 = self.c2f_3(y2)

        y3 = T.cat([self.conv_2(y2), x3], 1)
        y3 = self.c2f_4(y3)

        y1 = self.det_1(y1)
        y2 = self.det_2(y2)
        y3 = self.det_3(y3)

        return y1, y2, y3

    def compute_loss(self, y_pred, y):
        loss_fn = (
            T.nn.functional.smooth_l1_loss
            if self.continuous_output
            else T.nn.functional.cross_entropy
        )
        return loss_fn(y_pred, y)


class PyramidTransformerHead(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(
        self,
        input_dims: int,
        input_length: int,
        output_dims: int,
        output_length: int,
        num_heads: int,
    ):
        super().__init__()

        self.output_shape = (output_length, output_dims)

        self.upsample_2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_dims, input_dims // 2, 1, 1),
        )
        self.pyramid_transformer_2 = PyramidTransformer(
            input_dims // 2, input_length * 4, num_heads
        )

        self.upsample_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_dims // 2, input_dims // 4, 1, 1),
        )
        self.pyramid_transformer_1 = PyramidTransformer(
            input_dims // 4, input_length * 16, num_heads
        )

        self.memory_encoder_3 = nn.Sequential(
            C2f(input_dims, output_dims, 1, shortcut=True), nn.Flatten(2)
        )
        self.memory_encoder_2 = nn.Sequential(
            C2f(input_dims // 2, output_dims, 1, shortcut=True), nn.Flatten(2)
        )
        self.memory_encoder_1 = nn.Sequential(
            C2f(input_dims // 4, output_dims, 1, shortcut=True), nn.Flatten(2)
        )

        self.ll_lm_attention_3 = LL_LM_Attention(
            output_dims, output_length, input_length, num_heads
        )
        self.ll_lm_attention_2 = LL_LM_Attention(
            output_dims, output_length, input_length * 4, num_heads
        )
        self.ll_lm_attention_1 = LL_LM_Attention(
            output_dims, output_length, input_length * 16, num_heads
        )

        self.initial_predictor = nn.Sequential(
            SPPF(input_dims, input_dims),
            C2f(input_dims, input_dims, 2, shortcut=True),
            C2f(input_dims, output_dims * output_length, 2, shortcut=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, x):
        z_1, z_2, z_3 = x

        memory_3 = z_3

        memory_2 = self.pyramid_transformer_2(
            feature_map=z_2, memory=self.upsample_2(memory_3)
        )

        memory_1 = self.pyramid_transformer_1(
            feature_map=z_1, memory=self.upsample_1(memory_2)
        )

        enc_mem_3 = self.memory_encoder_3(memory_3).permute(0, 2, 1)
        enc_mem_2 = self.memory_encoder_2(memory_2).permute(0, 2, 1)
        enc_mem_1 = self.memory_encoder_1(memory_1).permute(0, 2, 1)

        pred_3 = self.initial_predictor(memory_3).reshape(-1, *self.output_shape)
        pred_2 = self.ll_lm_attention_3(pred_3, enc_mem_3) + pred_3
        pred_1 = self.ll_lm_attention_2(pred_2, enc_mem_2) + pred_2
        pred_0 = self.ll_lm_attention_1(pred_1, enc_mem_1) + pred_1

        return pred_3, pred_2, pred_1, pred_0

    def compute_loss(self, y_pred, y, loss_fn=F.l1_loss, **loss_fn_kwargs):
        # compute residual loss for each of the predictions
        return [loss_fn(y_p, y, **loss_fn_kwargs) for y_p in y_pred]


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        num_codebook_entries: int,
        output_dim: int,
        codebook_sequence_length: int,
        output_sequence_length: int,
        transformer_kwargs: Kwargs,
    ):
        super(type(self), self).__init__()

        self.input_state = nn.Parameter(
            T.randn(codebook_sequence_length, codebook_dim, dtype=T.float32),
            requires_grad=True,
        )
        self.codebook = nn.Embedding(num_codebook_entries, codebook_dim)
        self.output_state = nn.Parameter(
            T.randn(output_sequence_length, codebook_dim, dtype=T.float32),
            requires_grad=True,
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim + codebook_dim) // 2),
            nn.LayerNorm((input_dim + codebook_dim) // 2),
            nn.SiLU(),
            nn.Linear((input_dim + codebook_dim) // 2, codebook_dim),
            nn.LayerNorm(codebook_dim),
            nn.SiLU(),
        )

        self.tf_encoder = nn.Transformer(
            d_model=codebook_dim, **transformer_kwargs, batch_first=True
        )
        self.tf_decoder = nn.Transformer(
            d_model=codebook_dim, **transformer_kwargs, batch_first=True
        )

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

        z = self.tf_encoder(src=x, tgt=self.input_state.expand(x.shape[0], -1, -1))

        c = T.cdist(z, self.codebook.weight, p=2).argmin(-1)
        z_ = self.codebook(c)

        rec = None
        if self.train:
            rec = self.tf_decoder(
                src=z_, tgt=self.output_state.expand(z.shape[0], -1, -1)
            )
            rec = self.decoder(rec)

        return {
            "z": z,
            "discrete_z": z_,
            "codebook_indecies": c,
            "codebook_onehots": nn.functional.one_hot(
                c, self.codebook.num_embeddings
            ).to(dtype=T.float32),
            "reconstruction": rec,
        }

    def compute_loss(self, y, y_pred, z, codebook_indecies):
        loss_fn = T.nn.functional.l1_loss
        return loss_fn(y_pred, y), loss_fn(z, self.codebook(codebook_indecies))
