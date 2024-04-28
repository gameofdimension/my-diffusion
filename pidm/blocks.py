import torch
from torch import nn


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class Upsample(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return x


class Downsample(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        stride = 2
        self.op = torch.nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float,
        up: bool,
        down: bool,
        has_lateral=False,
    ) -> None:
        super().__init__()
        self.has_lateral = has_lateral
        self.out_channels = out_channels
        layers = [
            torch.nn.GroupNorm(32, in_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        ]
        self.in_layers = nn.Sequential(*layers)

        if up:
            self.h_upd = Upsample(in_channels)
            self.x_upd = Upsample(in_channels)
        elif down:
            self.h_upd = Downsample(in_channels)
            self.x_upd = Downsample(in_channels)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.updown = up or down

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels),
        )

        layers = [
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        ]
        self.out_layers = nn.Sequential(*layers)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, 1, padding=0)

    def forward(self, x, emb, lateral=None):
        if self.has_lateral:
            assert lateral is not None
            x = torch.cat([x, lateral], dim=1)

        if self.updown:
            h = self.in_layers[:-1](x)
            h = self.h_upd(h)
            h = self.in_layers[-1](h)
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        new_size = tuple(emb_out.shape) + (1,)*(h.ndim-emb_out.ndim)
        emb_out = emb_out.reshape(new_size)

        h = self.out_layers[0](h)
        if emb_out.shape[1] == self.out_channels*2:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale)
            h = h+shift
        else:
            scale = emb_out
            h = h * (1 + scale)
        h = self.out_layers[1:](h)

        x = self.skip_connection(x)
        return x + h


class EncoderResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        down: bool,
    ):
        super().__init__()
        layers = [
            torch.nn.GroupNorm(32, in_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        ]
        self.in_layers = nn.Sequential(*layers)

        if down:
            self.h_upd = Downsample(in_channels)
            self.x_upd = Downsample(in_channels)
        else:
            self.h_upd = self.x_upd = torch.nn.Identity()
        self.down = down

        layers = [
            torch.nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        ]
        self.out_layers = nn.Sequential(*layers)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, 1, padding=0)

    def forward(self, x):
        if self.down:
            h = self.in_layers[:-1](x)
            h = self.h_upd(h)
            h = self.in_layers[-1](h)
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
    ):
        super().__init__()
        self.channels = channels
        num_heads = 1
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.to_kv = nn.Conv1d(channels, channels * 2, 1)
        self.to_q = nn.Conv1d(channels, channels * 1, 1)
        self.selfattention = QKVAttention(self.num_heads)
        self.crossattention = QKVAttention(self.num_heads)

        self.proj_out2 = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, cond):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        cond = cond.reshape(b, c, -1)
        kv = self.to_kv(cond)
        q = self.to_q(self.norm(x))
        qkv = torch.cat([q, kv], 1)
        h = self.crossattention(qkv)
        h = self.proj_out2(h)

        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(
            bs * self.n_heads, ch * 3, length
        ).split(ch, dim=1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(bs, -1, length)
