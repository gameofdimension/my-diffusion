import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import UpBlock2D as GoldUpBlock2D
from diffusers.models.upsampling import Upsample2D as GoldUpsample2D

from sd.blocks import UpBlock2D, Upsample2D


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

    def test_downblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldUpBlock2D):
                print(mod[0])
                gold_block = mod[1]
                channels = 1280
                block = UpBlock2D(
                    in_channels=channels,
                    out_channels=channels,
                    prev_output_channel=channels,
                    temb_channels=channels,
                )
                block.load_state_dict(gold_block.state_dict())

                data = torch.randn(10, 1280, 8, 8)
                temb = torch.randn(10, 1280)
                res_hidden_states_tuple = [
                    torch.randn(10, 1280, 8, 8),
                    torch.randn(10, 1280, 8, 8),
                    torch.randn(10, 1280, 8, 8),
                ]
                output = block(
                    hidden_states=data, temb=temb,
                    res_hidden_states_tuple=res_hidden_states_tuple)
                gold = gold_block(
                    hidden_states=data, temb=temb,
                    res_hidden_states_tuple=res_hidden_states_tuple)

                delta = (gold - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6
