import torch
from torch import nn

from uvit.blocks import Block
from uvit.embedding import PatchEmbed, TimestepEmbed


def unpatchify(x, channels):
    b, l, c = x.shape
    patch_size = int((c // channels) ** 0.5)
    h = w = int(l ** 0.5)
    assert h*w == l
    x = x.reshape(b, h, w, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(
        b, channels, patch_size*h, patch_size*w)
    return x


class Model(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        clip_dim,
        num_clip_token,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.context_embed = nn.Linear(clip_dim, embed_dim)
        self.time_embed = TimestepEmbed(dim=embed_dim)

        self.extras = 1 + num_clip_token
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, skip=False)
            for _ in range(depth // 2)])
        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, skip=False)
        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, skip=True)
            for _ in range(depth // 2)])

        self.norm = nn.LayerNorm(embed_dim)
        self.out_chans = in_chans
        patch_dim = patch_size ** 2 * self.out_chans
        self.decoder_pred = nn.Linear(embed_dim, patch_dim, bias=True)
        self.final_layer = nn.Conv2d(
            self.out_chans, self.out_chans, 3, padding=1)

    def forward(self, x, t, c):
        x = self.patch_embed(x)
        L = x.size(1)
        c = self.context_embed(c)
        t = self.time_embed(t).unsqueeze(1)

        x = torch.cat((t, c, x), dim=1)
        x = x+self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.out_chans)
        x = self.final_layer(x)
        return x
