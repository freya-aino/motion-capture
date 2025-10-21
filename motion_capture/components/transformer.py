import torch as T
import torch.nn as nn

from motion_capture.utils.torchhelpers import positionalencoding1d


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x + 1


class AttentionBlock(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(self, input_dims: int, num_heads: int):
        super().__init__()

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=input_dims, num_heads=num_heads, batch_first=True
        )
        self.attention_norm_1 = nn.LayerNorm(input_dims)
        self.attention_ff = nn.Sequential(nn.Linear(input_dims, input_dims), nn.SiLU())
        self.attention_norm_2 = nn.LayerNorm(input_dims)

    def forward(self, q, k, v):
        z_hat, _ = self.multi_head_attention(query=q, key=k, value=v)
        z_hat = self.attention_norm_1(z_hat + q)
        y = self.attention_norm_2(self.attention_ff(z_hat) + z_hat)
        return y


class PyramidTransformer(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(self, input_dims: int, input_length: int, num_heads: int):
        super().__init__()

        self.attention_block = AttentionBlock(input_dims, num_heads)
        self.positional_encoding = nn.Parameter(
            positionalencoding1d(make_even(input_length), make_even(input_dims))[
                :input_dims, :input_length
            ],
            requires_grad=False,
        )

    def forward(self, feature_map, memory):
        in_shape = feature_map.shape

        v_hat = (feature_map * memory).flatten(2)
        pos_enc = self.positional_encoding.expand(feature_map.shape[0], -1, -1)
        feature_map = feature_map.flatten(2)

        y = self.attention_block(
            q=(feature_map + pos_enc).permute(0, 2, 1),
            k=(v_hat + pos_enc).permute(0, 2, 1),
            v=v_hat.permute(0, 2, 1),
        )

        return y.permute(0, 2, 1).reshape(in_shape)


class LL_LM_Attention(nn.Module):
    # based on: https://arxiv.org/pdf/2207.03917.pdf
    def __init__(
        self, input_dims: int, query_length: int, memory_length: int, num_heads: int
    ):
        super().__init__()

        self.query_positional_encoding = nn.Parameter(
            positionalencoding1d(make_even(query_length), make_even(input_dims))[
                :input_dims, :query_length
            ].T,
            requires_grad=False,
        )
        self.memory_positional_encoding = nn.Parameter(
            positionalencoding1d(make_even(memory_length), make_even(input_dims))[
                :input_dims, :memory_length
            ].T,
            requires_grad=False,
        )

        self.ll_attention_block = AttentionBlock(input_dims, 1)
        self.lm_attention_block = AttentionBlock(input_dims, 1)

    def forward(self, queries, memory):
        qk = queries + self.query_positional_encoding.expand(queries.shape[0], -1, -1)

        queries = self.ll_attention_block(qk, qk, queries)

        lm_q = queries + self.query_positional_encoding.expand(queries.shape[0], -1, -1)
        lm_k = memory + self.memory_positional_encoding.expand(memory.shape[0], -1, -1)

        queries = self.lm_attention_block(lm_q, lm_k, memory)

        return queries
