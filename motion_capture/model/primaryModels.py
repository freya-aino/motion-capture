import time
import torch as T
import torch.nn as nn

from .convolution.backbones import Backbone
from .convolution.necks import UpsampleCrossAttentionrNeck
from .convolution.heads import SimpleTransformerHead


class FullConvModel(nn.Module):
    """
        Full model with convolutional backbone, neck and head.
        
        the backbone is expected to output a list of tensors, from different depths of the network.
    """
    
    def __init__(
        self, 
        neck_output_size,
        head_output_size,
        head_output_length,
        depth_multiple = 0.33,
        width_multiple = 0.25):
        
        super(type(self), self).__init__()
        
        self.backbone = Backbone(depth_multiple, width_multiple)
        self.neck = UpsampleCrossAttentionrNeck(neck_output_size, depth_multiple=depth_multiple, width_multiple=width_multiple)
        
        self.head = SimpleTransformerHead(neck_output_size, head_output_size, head_output_length, width_multiple=width_multiple)
        
        print(f"running with neck: {self.neck.__class__}")
        
    def forward(self, x: T.Tensor):
        o = self.backbone(x)
        o = self.neck(*o)
        o = self.head(o)
        return o


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

