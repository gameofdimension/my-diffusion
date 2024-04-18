import torch
from diffusers.models.embeddings import Timesteps as GoldTimesteps
from diffusers.models.embeddings import TimestepEmbedding as GoldTransfer

from sd.time_embed import TimestepEmbedding, Timesteps


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
