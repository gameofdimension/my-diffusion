import math

import numpy as np
import torch
from torch import nn


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: int,
            patch_size: int,
            in_chans: int,
            embed_dim: int,
            bias: bool,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = tuple(
            [s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.in_chans = in_chans
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size,
            stride=patch_size, bias=bias)
        self.norm = nn.Identity()

    def forward(self, x):
        _, C, H, W = x.shape

        assert C == self.in_chans
        assert H == self.img_size[0]
        assert W == self.img_size[1]
        assert H % self.patch_size[0] == 0
        assert W % self.patch_size[1] == 0
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


class LabelEmbed(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # 0-999 for normal classes, 1000 for null class
        self.embedding_table = nn.Embedding(
            num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        """
        Drops labels to enable classifier-free guidance.
        """
        drop_ids = torch.rand(
            labels.shape[0], device=labels.device) < self.dropout_prob
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout):
            labels = self.token_drop(labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size):
        frequency_embedding_size = 256
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        max_period = 10000
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_2d_pos_embed(embed_dim, grid_size):

    def make_table(max_len, dim):
        pos = np.arange(max_len, dtype=np.float64)
        max_period = 10000
        assert dim % 2 == 0
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(
                start=0, stop=half, dtype=np.float64) / half
        )
        args = pos[:, None] * freqs[None]
        embedding = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        return embedding

    assert embed_dim % 2 == 0
    pos_embed = make_table(grid_size, embed_dim//2)
    embedding = np.zeros((grid_size*grid_size, embed_dim))
    for r in range(grid_size):
        for c in range(grid_size):
            w_embedding = pos_embed[c]
            h_embedding = pos_embed[r]
            embedding[r*grid_size+c] = np.concatenate(
                (w_embedding, h_embedding), axis=-1)

    return embedding
