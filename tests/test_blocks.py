import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.downsampling import Downsample2D as GoldDownsample2D
from diffusers.models.resnet import ResnetBlock2D as GoldResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import \
    CrossAttnDownBlock2D as GoldAttnDownBlock2D
from diffusers.models.unets.unet_2d_blocks import \
    DownBlock2D as GoldDownBlock2D

from sd.blocks import (CrossAttnDownBlock2D, DownBlock2D, Downsample2D,
                       ResnetBlock2D)


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

    def test_downsample(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldDownsample2D):
                gold_block = mod[1]
                block = Downsample2D(
                    channels=gold_block.channels,
                    out_channels=gold_block.out_channels,
                    padding=gold_block.padding,
                    name=gold_block.name,
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

    def test_resnetblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldResnetBlock2D):
                gold_block = mod[1]
                resnet = ResnetBlock2D(
                    in_channels=gold_block.in_channels,
                    out_channels=gold_block.out_channels,
                    temb_channels=1280,
                )
                resnet.load_state_dict(gold_block.state_dict())

                data = torch.randn(10, gold_block.in_channels, 16, 16)
                temb = torch.randn(10, 1280)
                gold = gold_block(data, temb=temb)
                output = resnet(data, temb=temb)

                delta = (gold - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6

    def test_downblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldDownBlock2D):
                print(mod[0])
                gold_block = mod[1]
                block = DownBlock2D(
                    in_channels=1280,
                    out_channels=1280,
                    temb_channels=1280,
                    num_layers=2,
                )
                block.load_state_dict(gold_block.state_dict())

                data = torch.randn(10, 1280, 16, 16)
                temb = torch.randn(10, 1280)
                output = block(hidden_states=data, temb=temb)[0]
                gold = gold_block(hidden_states=data, temb=temb)[0]

                delta = (gold - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6

    def test_crossattnblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldAttnDownBlock2D):
                gold_block = mod[1]

                num_attention_heads = gold_block.num_attention_heads
                if num_attention_heads == 5:
                    in_channels = 320
                    out_channels = 320
                elif num_attention_heads == 10:
                    in_channels = 320
                    out_channels = 640
                elif num_attention_heads == 20:
                    in_channels = 640
                    out_channels = 1280
                else:
                    assert False

                temb_channels = 1280
                block = CrossAttnDownBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_layers=2,
                    transformer_layers_per_block=[1, 1],
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=1024,
                    use_linear_projection=True,
                    add_downsample=gold_block.downsamplers is not None,
                )
                block.load_state_dict(gold_block.state_dict())

                h = 1280*16//out_channels
                hidden_states = torch.randn(40, in_channels, h, h)
                temb = torch.randn(40, temb_channels)
                encoder_hidden_states = torch.randn(40, 77, 1024)
                gold = gold_block(
                    hidden_states=hidden_states, temb=temb,
                    encoder_hidden_states=encoder_hidden_states)[0]
                output = block(
                    hidden_states=hidden_states, temb=temb,
                    encoder_hidden_states=encoder_hidden_states)[0]

                delta = (gold - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6
