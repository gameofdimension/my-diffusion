from typing import List

import torch

from pidm.blocks import EncoderResBlock


class Encoder(torch.nn.Module):
    def __init__(
            self, in_channels: int, model_channels: int,
            channel_mult: List[int], dropout: float) -> None:
        super().__init__()
        ch = channel_mult[0]*model_channels
        self.input_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(in_channels, ch, 3, padding=1))
        ])

        num_res_blocks = 2
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                resblock = EncoderResBlock(
                    in_channels=ch,
                    out_channels=mult*model_channels,
                    dropout=dropout,
                    down=False,
                )
                ch = mult*model_channels
                self.input_blocks.append(torch.nn.Sequential(resblock))

            if level == len(channel_mult) - 1:
                continue

            resdownblock = EncoderResBlock(
                in_channels=ch,
                out_channels=ch,
                dropout=dropout,
                down=True,
            )
            self.input_blocks.append(torch.nn.Sequential(resdownblock))

    def forward(self, x):
        h = x
        results = []
        for block in self.input_blocks:
            h = block(h)
            results.append(h)
        return results
