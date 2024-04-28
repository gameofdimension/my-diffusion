
from typing import List

import torch
from torch import nn

from pidm.blocks import AttentionBlock, ResBlock, zero_module
from pidm.encoder import Encoder
from pidm.time_embed import TimeEmbed, timestep_embedding


def run_block(block, h, emb, cond, lateral=None):
    for layer in block:
        if isinstance(layer, ResBlock):
            h = layer(h, emb=emb, lateral=lateral)
        elif isinstance(layer, AttentionBlock):
            h = layer(h, cond=cond)
        else:
            h = layer(h)
    return h


class Model(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        embed_channels: int,
        channel_mult: List[int],
        attention_resolutions: List[int],
        dropout: float,
    ) -> None:
        super().__init__()

        self.model_channels = model_channels
        self.channel_mult = channel_mult
        ch = input_ch = channel_mult[0] * model_channels
        self.input_blocks = nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(in_channels, ch, 3, padding=1))
        ])

        input_block_chans = [[] for _ in range(len(channel_mult))]
        input_block_chans[0].append(ch)
        self.input_num_blocks = [0 for _ in range(len(self.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(self.channel_mult))]

        num_res_blocks = 2
        resolution = image_size
        for level, mult in enumerate(channel_mult):
            for block_id in range(num_res_blocks):
                layers: list = [
                    ResBlock(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        emb_channels=embed_channels,
                        dropout=dropout,
                        up=False,
                        down=False,
                    )
                ]
                ch = mult * model_channels
                if (resolution in attention_resolutions
                        and block_id == num_res_blocks-1):
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
            if level != len(channel_mult) - 1:
                resolution //= 2
                resdownblock = ResBlock(
                    in_channels=ch,
                    out_channels=ch,
                    emb_channels=embed_channels,
                    dropout=dropout,
                    up=False,
                    down=True,
                )
                self.input_blocks.append(torch.nn.Sequential(resdownblock))
                input_block_chans[level+1].append(ch)
                self.input_num_blocks[level + 1] += 1

        self.middle_block = nn.Sequential(
            ResBlock(
                in_channels=ch,
                out_channels=ch,
                emb_channels=embed_channels,
                dropout=dropout,
                up=False,
                down=False,
            ),
            AttentionBlock(ch),
            ResBlock(
                in_channels=ch,
                out_channels=ch,
                emb_channels=embed_channels,
                dropout=dropout,
                up=False,
                down=False,
            ),
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans[level].pop()
                layers = [
                    ResBlock(
                        in_channels=ch+ich,
                        out_channels=model_channels*mult,
                        emb_channels=embed_channels,
                        dropout=dropout,
                        up=False,
                        down=False,
                        has_lateral=True,
                    )
                ]
                ch = model_channels * mult
                if (resolution in attention_resolutions
                        and i == num_res_blocks-1):
                    layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    resolution *= 2
                    resupblock = ResBlock(
                        in_channels=ch,
                        out_channels=ch,
                        emb_channels=embed_channels,
                        dropout=dropout,
                        up=True,
                        down=False,
                    )
                    layers.append(resupblock)
                self.output_blocks.append(nn.Sequential(*layers))
                self.output_num_blocks[level] += 1

        self.time_embed = TimeEmbed(
            time_channels=model_channels,
            time_out_channels=embed_channels,
        )
        self.encoder = Encoder(
            3, model_channels, channel_mult, dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, t, ref):
        t_emb = timestep_embedding(t, self.model_channels)
        emb = self.time_embed(t_emb)
        cond_lst = self.encoder(ref)
        hs = [[] for _ in range(len(self.channel_mult))]

        k = 0
        h = x
        for i in range(len(self.input_num_blocks)):
            for _ in range(self.input_num_blocks[i]):
                h = run_block(
                    self.input_blocks[k],
                    h=h,
                    emb=emb,
                    cond=cond_lst[k]
                )
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = run_block(self.middle_block, h=h, emb=emb, cond=cond_lst[-1])

        k = 0
        for i in range(len(self.output_num_blocks)):
            for _ in range(self.output_num_blocks[i]):
                lateral = hs[-i - 1].pop()
                h = run_block(
                    self.output_blocks[k],
                    h=h,
                    emb=emb,
                    cond=cond_lst[-k-1],
                    lateral=lateral
                )
                k += 1

        pred = self.out(h)
        return pred
