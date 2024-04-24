import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import \
    CrossAttnUpBlock2D as GoldAttnUpBlock2D
from diffusers.models.unets.unet_2d_blocks import \
    UNetMidBlock2DCrossAttn as GoldUNetMidBlock2DCrossAttn
from diffusers.models.unets.unet_2d_blocks import UpBlock2D as GoldUpBlock2D
from diffusers.models.upsampling import Upsample2D as GoldUpsample2D

from sd.blocks import (CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D,
                       Upsample2D)


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
                assert delta < 1e-6

    def test_downblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldUpBlock2D):
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
                assert delta < 1e-6

    def test_crossattnupblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldAttnUpBlock2D):
                gold_block = mod[1]

                num_attention_heads = gold_block.num_attention_heads
                if num_attention_heads == 20:
                    in_channels = 640
                    out_channels = 1280
                    prev_output_channel = 1280
                elif num_attention_heads == 10:
                    in_channels = 320
                    out_channels = 640
                    prev_output_channel = 1280
                elif num_attention_heads == 5:
                    in_channels = 320
                    out_channels = 320
                    prev_output_channel = 640
                else:
                    assert False

                block = CrossAttnUpBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    prev_output_channel=prev_output_channel,
                    temb_channels=1280,
                    num_layers=3,
                    transformer_layers_per_block=[1, 1, 1],
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=1024,
                    use_linear_projection=True,
                    add_upsample=gold_block.upsamplers is not None,
                )
                block.load_state_dict(gold_block.state_dict())

                temb = torch.randn(10, 1280)
                encoder_hidden_states = torch.randn(10, 77, 1024)
                if num_attention_heads == 20:
                    hidden_states = torch.randn(10, 1280, 16, 16)
                    res_hidden_states_tuple = [
                        torch.randn(10, 640, 16, 16),
                        torch.randn(10, 1280, 16, 16),
                        torch.randn(10, 1280, 16, 16),
                    ]
                elif num_attention_heads == 10:
                    hidden_states = torch.randn(10, 1280, 32, 32)
                    res_hidden_states_tuple = [
                        torch.randn(10, 320, 32, 32),
                        torch.randn(10, 640, 32, 32),
                        torch.randn(10, 640, 32, 32),
                    ]
                elif num_attention_heads == 5:
                    hidden_states = torch.randn(10, 640, 64, 64)
                    res_hidden_states_tuple = [
                        torch.randn(10, 320, 64, 64),
                        torch.randn(10, 320, 64, 64),
                        torch.randn(10, 320, 64, 64),
                    ]
                else:
                    assert False

                gold = gold_block(
                    hidden_states=hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_hidden_states_tuple,
                )
                output = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_hidden_states_tuple,
                )
                delta = (gold - output).abs().max().item()
                assert delta < 1e-6

    def test_middleblock(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldUNetMidBlock2DCrossAttn):
                gold_block = mod[1]

                block = UNetMidBlock2DCrossAttn(
                    in_channels=1280,
                    temb_channels=1280,
                    num_layers=1,
                    transformer_layers_per_block=[1],
                    num_attention_heads=gold_block.num_attention_heads,
                    cross_attention_dim=1024,
                    use_linear_projection=True,
                )
                block.load_state_dict(gold_block.state_dict())

                temb = torch.randn(10, 1280)
                encoder_hidden_states = torch.randn(10, 77, 1024)
                hidden_states = torch.randn(10, 1280, 8, 8)

                gold = gold_block(
                    hidden_states=hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                )
                output = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                )
                delta = (gold - output).abs().max().item()
                assert delta < 1e-6
