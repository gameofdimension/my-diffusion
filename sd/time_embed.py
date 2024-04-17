from diffusers.models.embeddings import Timesteps as GoldTimesteps
from diffusers.models.embeddings import TimestepEmbedding as GoldTransfer
import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
    ):
        super().__init__()
        act_fn: str = "silu"
        sample_proj_bias = True

        self.linear_1 = nn.Linear(
            in_channels, time_embed_dim, sample_proj_bias)
        assert act_fn == "silu"
        self.act = nn.SiLU()
        time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(
            time_embed_dim, time_embed_dim_out, sample_proj_bias)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic
    Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(torch.nn.Module):
    def __init__(
            self, num_channels: int, flip_sin_to_cos: bool,
            downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


def test_embedding():
    torch.manual_seed(0)

    timesteps = torch.randint(0, 1000, (10,))
    mod_timesteps = Timesteps(
        320, flip_sin_to_cos=True, downscale_freq_shift=0)
    embeddings = mod_timesteps(timesteps)

    gold_mod = GoldTimesteps(
        320, flip_sin_to_cos=True, downscale_freq_shift=0)
    gold_embeddings = gold_mod(timesteps)

    print((embeddings-gold_embeddings).abs().max())


def test_transform_embedding():
    in_channels = 320
    out_channels = 320*4

    weight1 = torch.randn(out_channels, in_channels)
    bias1 = torch.randn(out_channels)

    weight2 = torch.randn(out_channels, out_channels)
    bias2 = torch.randn(out_channels)

    mod = TimestepEmbedding(in_channels, out_channels)
    mod.linear_1.weight.data.copy_(weight1)
    mod.linear_1.bias.data.copy_(bias1)
    mod.linear_2.weight.data.copy_(weight2)
    mod.linear_2.bias.data.copy_(bias2)

    gold_mod = GoldTransfer(in_channels, out_channels)
    gold_mod.linear_1.weight.data.copy_(weight1)
    gold_mod.linear_1.bias.data.copy_(bias1)
    gold_mod.linear_2.weight.data.copy_(weight2)
    gold_mod.linear_2.bias.data.copy_(bias2)

    data = torch.randn(10, in_channels)
    out = mod(data)
    gold = gold_mod(data)

    print((out-gold).abs().max())


if __name__ == "__main__":
    # test_embedding()
    test_transform_embedding()
