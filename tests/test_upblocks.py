import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.upsampling import Upsample2D as GoldUpsample2D

from sd.blocks import (Upsample2D)


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        checkpoint = 'stabilityai/stable-diffusion-2-1'
        cls.unet = UNet2DConditionModel.from_pretrained(
            checkpoint, subfolder="unet",
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def test_upsample(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldUpsample2D):
                gold_block = mod[1]
                block = Upsample2D(
                    channels=gold_block.channels,
                    out_channels=gold_block.out_channels,
                )
                block.load_state_dict(gold_block.state_dict())

                in_channels = gold_block.channels
                h = 320*64//in_channels
                hidden_states = torch.randn(40, in_channels, h, h)
                gold = gold_block(hidden_states)
                output = block(hidden_states)

                delta = (gold - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6
