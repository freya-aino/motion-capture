from typing import List
import torch as T
import torch.nn as nn

from .transformer import TransformerEncoderBlock
from ..core.torchhelpers import positional_embedding


class VQVAE(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        num_codebook_entries: int,
        codebook_dim: int,
        encoder_sequence_length: int,
        depth: int,
        ):
        
        super(type(self), self).__init__()
        
        # self.internal_state = nn.Parameter(T.randn(encoder_sequence_length, input_dim, dtype=T.float32), requires_grad=True)
        self.positional_encoding = nn.Parameter(positional_embedding(encoder_sequence_length, input_dim), requires_grad=False)
        self.encoder = TransformerEncoderBlock(input_dim, codebook_dim, depth=depth)
        self.codebook = nn.Embedding(num_codebook_entries, codebook_dim)
        self.decoder = TransformerEncoderBlock(codebook_dim, output_dim, depth=depth)
        
        
    def forward(self, x: T.Tensor):
        # expects inputs of shape: batch_size, dims, sequence_lengths ...
        
        z1 = x.flatten(2).permute(0, 2, 1)
        internal_state = self.internal_state[z1.shape[1]:].expand(z1.shape[0], -1, -1)
        
        if z1.shape[1] > self.internal_state.shape[0]:
            print("Warning: encoder sequence length is longer than internal state length, truncating")
        
        # z2 = T.cat([z1[:, :self.internal_state.shape[0]], internal_state], 1)
        z2 = z2 + self.positional_encoding.expand(z2.shape[0], -1, -1)
        
        z = self.encoder(z2) # non discrete
        c = T.cdist(z, self.codebook.weight, p = 2).argmin(-1)
        z_ = self.codebook(c)
        
        rec = None
        if self.train:
            rec = self.decoder(z_).permute(0, 2, 1)
        
        return {
            "z": z,
            "discrete_z": z_,
            "codebook_indecies": c,
            "reconstruction": rec
        }
    
    def compute_loss(self, y, y_pred, z, codebook_indecies, alpha = 1, beta = 0.25):
        loss_fn = T.nn.functional.l1_loss
        return alpha * loss_fn(y_pred, y) + beta * loss_fn(z, self.codebook(codebook_indecies))





# class SpatialEncoder(nn.Module):
#     def __init__(self, backbone: nn.Module, config: dict):
#         super(type(self), self).__init__()

#         total_number_of_chunks = config["image_encoder"]["num_chunks"][0] * config["image_encoder"]["num_chunks"][1]

#         self.image_encoder = ChunkImageEncoder(backbone=backbone, **config["image_encoder"])
#         self.spatial_transformer = SpatialTransformer(total_number_of_chunks=total_number_of_chunks, **config["spatial_transformer"])
        
#     def forward(self, x: T.Tensor):

#         x = self.image_encoder(x)
#         x = self.spatial_transformer(x)

#         return x

# class TemporalEncoder(nn.Module):
#     def __init__(self, rnn_energy_net: RNNEnergyNet, temporal_transformer: TemporalTransformer):
#         super(type(self), self).__init__()

#         self.rnn_energy_net = rnn_energy_net
#         self.temporal_transformer = temporal_transformer

#     def forward(self, x: T.Tensor):
#         y_rnn_energy_net, y_p_rnn_energy_net = self.rnn_energy_net(x)

#         energies = self.rnn_energy_net.energy_function(y_rnn_energy_net, y_p_rnn_energy_net)
#         energies = energies.mean(dim = -1, keepdim = True)

#         y_temporal_transformer = self.temporal_transformer(T.cat([y_rnn_energy_net, energies.detach()], dim = -1))
#         return (y_rnn_energy_net, y_p_rnn_energy_net, energies), y_temporal_transformer



# class VisualSpatioTemporalDiscriminativeTransformerEnergyNetwork(nn.Module):
    
#     def __init__(self, backbone: nn.Module, model_config: dict):
#         super(type(self), self).__init__()

#         assert model_config["image_encoder"]["output_size"] == model_config["spatial_transformer"]["input_size"], "image encoder and spatial transformer input and output size need to match"

#         self.spatial_encoder = SpatialEncoder(backbone, model_config["image_encoder"])

#         rnn_energy_net = RNNEnergyNet(**model_config["rnn_energy_net"])
#         temporal_transformer = TemporalTransformer(**model_config["temporal_transformer"], max_sequence_length=model_config["max_temporal_sequence_length"])
#         self.temporal_encoder = TemporalEncoder(rnn_energy_net, temporal_transformer)

#         self.spatiotemporal_discriminator = SpatioTemporalDiscriminator(**model_config["spatio_temporal_discriminator"], max_sequence_length=model_config["max_temporal_sequence_length"])

#         self.max_temporal_sequence_length = model_config["max_temporal_sequence_length"]
#         self.encoded_frame_buffer = []


#     def forward(self, x: T.Tensor, output_masks: list = [], train_spatial = False):

#         # if train_spatial:
#         #     x = self.spatial_encoder(x)
#         # else:
#         #     with T.inference_mode():
#         #         x = self.spatial_encoder(x)

#         #     self.encoded_frame_buffer.append(x)
#         #     if len(self.encoded_frame_buffer) > self.max_temporal_sequence_length:
#         #         self.encoded_frame_buffer = self.encoded_frame_buffer[1:]
#         #     x = T.cat(self.encoded_frame_buffer, dim = 1)
        
#         # ---------------------------------------------

#         (E_y, E_y_p, E_energies), y_hat = self.temporal_encoder(x)
#         y_discriminator = [self.spatiotemporal_discriminator(y_hat, om) for om in output_masks]

#         return {
#             "prediction": y_hat,
#             "discriminator_samples": y_discriminator,
#             "rnn_energy_net_y": E_y,
#             "rnn_energy_net_y_p": E_y_p,
#             "rnn_energy_net_energies": E_energies,
#         }

#     def reset_sequence(self):
#         self.encoded_frame_buffer = []



class RNNEnergyNet(nn.Module):
    def __init__(
        self, 
        input_size: int,
        latent_size: int,
        output_size: int,
        z_size: int,
        optimal_variance: int,
        energy_function=nn.L1Loss(reduction="none")):
        super(type(self), self).__init__()
        
        self.encoder = nn.GRU(input_size, latent_size, batch_first = True)
        self.encoder_head = nn.Sequential(
            nn.Linear(latent_size, (latent_size + output_size) // 2),
            nn.BatchNorm1d((latent_size + output_size) // 2),
            nn.ELU(),
            nn.Linear((latent_size + output_size) // 2, output_size),
            nn.BatchNorm1d(output_size),
            nn.ELU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(output_size + z_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ELU(),
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ELU()
        )
        
        self.Z = nn.Parameter(T.randn(z_size, dtype=T.float32, requires_grad = True))
        self.optimal_variance = nn.Parameter(T.tensor([optimal_variance], dtype=T.float32, requires_grad = False))
        self.covar_identity = nn.Parameter(T.eye(output_size, dtype=T.float32, requires_grad = False))
        
        self.rand_mu = nn.Parameter(T.tensor([0], dtype = T.float32, requires_grad = False))
        self.rand_std = nn.Parameter(T.tensor([1], dtype = T.float32, requires_grad = False))
        self.random_sampler = T.distributions.Normal(self.rand_mu, self.rand_std)
        
        self.energy_function = energy_function
        
    def compute_loss(self, x: T.Tensor, x_p: T.Tensor):
        
        var = T.var(x, dim=-1)
        state_cov = T.cov(x.permute(2, 0, 1).flatten(1))
        
        return {
            "prediction_loss": self.energy_function(x[1:], x_p[:-1]),
            "covariance_loss": self.energy_function(state_cov, self.covar_identity),
            "variance_loss": self.energy_function(var, self.optimal_variance),
            "Z_loss": self.energy_function(self.Z, self.random_sampler.sample(self.Z.shape))
        }

    def forward(self, x: T.Tensor):
        
        assert len(x.shape) == 3, "input shape should be [B, S, Z]"
        
        bs, sl, _ = x.shape
        
        x = self.encoder(x)[0]
        x = x.reshape(bs * sl, -1)
        x = self.encoder_head(x)
        
        z = self.Z.expand(x.shape[0], self.Z.shape[0])
        x_p = self.predictor(T.cat([x, z], dim = -1))
        
        x = x.reshape(bs, sl, -1)
        x_p = x_p.reshape(bs, sl, -1)
        return x, x_p


class SpatioTemporalDiscriminator(nn.Module):
    def __init__(
        self,
        state_size : int,
        state_channels : int,
        max_sequence_length : int,
        transformer_latent_size : int,
        transformer_depth : int,
        energy_function=nn.L1Loss(reduction="none")):
        super(type(self), self).__init__()
        
        self.max_sequence_length = max_sequence_length
        self.state_size = state_size
        self.state_channels = state_channels 
        
        self.temporal_index = nn.Parameter(T.eye(max_sequence_length, dtype = T.float32, requires_grad = False))
        self.spatial_index = nn.Parameter(T.eye(state_size, dtype = T.float32, requires_grad = False))
        
        input_size = state_size + state_channels + max_sequence_length
        
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_size, 1, transformer_latent_size, batch_first = True), transformer_depth)
        self.encoder_head = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ELU(),
            nn.Linear(input_size // 2, state_channels),
            nn.BatchNorm1d(state_channels),
            nn.ELU(),
        )
        
        self.energy_function = energy_function
        
    def forward(self, x: T.Tensor, mask: T.Tensor):
        
        assert mask.dtype == T.bool, "mask has to be of type bool"
        assert len(x.shape) == 3, "input shape should be [B, S, Z]"
        assert x.shape[1] <= self.max_sequence_length, "max sequence length is " + str(self.max_sequence_length) + " got " + str(x.shape[1])
        
        bs, sl, _ = x.shape
        x = x.reshape(bs, sl, self.state_size, self.state_channels)
        _, _, z, c = x.shape
        
        xx = x
        
        mask = (mask * -T.inf).nan_to_num(0.)
        
        tmp_ind = self.temporal_index[:sl].expand(bs, z, -1, -1).permute(0, 2, 1, 3)
        spt_ind = self.spatial_index.expand(bs, sl, -1, -1)
        x = T.cat([x, tmp_ind, spt_ind], dim = -1)
        x = x.reshape(bs, sl * z, -1)
        
        x = self.encoder(x, mask)
        x = x.reshape(bs * sl * z, -1)
        x = self.encoder_head(x)
        x = x.reshape(bs, sl, z, c)
        
        e = self.energy_function(x, xx)
        
        return x, e
