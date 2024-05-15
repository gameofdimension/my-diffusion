import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias
    ):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        skip: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        qkv_bias = False
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = dim * 4
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
        )
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, skip=None):
        if skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
