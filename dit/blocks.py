import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act1 = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        qkv_bias = True
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = hidden_size * 4
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        assert c.dim() == 2
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=2)

        x = x + gate_msa * self.attn(
            (scale_msa+1)*self.norm1(x) + shift_msa)
        x = x + gate_mlp * self.mlp(
            (scale_mlp+1)*self.norm2(x) + shift_mlp)
        return x
