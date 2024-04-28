import math

import torch
from torch import nn


def timestep_embedding(timesteps, dim):
    max_period = 10000
    assert dim % 2 == 0
    half = dim // 2
    arange = torch.arange(start=0, end=half, dtype=torch.float32) / half
    freqs = torch.exp(
        -math.log(max_period) * arange
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class TimeEmbed(nn.Module):
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_channels, time_out_channels),
            nn.SiLU(),
            nn.Linear(time_out_channels, time_out_channels),
        )

    def forward(self, time_emb):
        time_emb = self.time_embed(time_emb)
        return time_emb
