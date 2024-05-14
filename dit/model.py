import torch
from torch import nn

from dit.blocks import Block
from dit.embedding import (LabelEmbed, PatchEmbed, TimestepEmbed,
                           get_2d_pos_embed)


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        in_channels,
        patch_size,
        num_heads,
        num_layers,
        class_dropout_prob=0.1
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True
        )
        self.t_embedder = TimestepEmbed(hidden_size)
        self.y_embedder = LabelEmbed(
            num_classes=1000,
            hidden_size=hidden_size,
            dropout_prob=class_dropout_prob
        )

        pos_embed = get_2d_pos_embed(
            embed_dim=hidden_size,
            grid_size=input_size//patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).unsqueeze(0).float(),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.final_layer = Head(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels
        )

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        n, s, d = x.shape
        assert d == self.patch_size ** 2 * self.out_channels
        c = self.out_channels
        p = self.patch_size
        h = w = int(s ** 0.5)
        assert h * w == s

        x = x.reshape(shape=(n, h, w, p, p, c))
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(shape=(n, c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        x = self.x_embedder(x)
        x = x + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        c = t+y

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


class Head(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        assert c.dim() == 2
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        x = (1+scale)*self.norm_final(x)+shift
        x = self.linear(x)
        return x
