import torch as T
import torch.nn as nn

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

        print(x.shape, mask.shape)

        x = self.encoder(x, mask)
        x = x.reshape(bs * sl * z, -1)
        x = self.encoder_head(x)
        x = x.reshape(bs, sl, z, c)

        e = self.energy_function(x, xx)

        return x, e
