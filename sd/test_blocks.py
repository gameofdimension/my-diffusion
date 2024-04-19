import torch
from diffusers import UNet2DConditionModel
from diffusers.models.downsampling import Downsample2D as GoldDownsample2D

from sd.blocks import Downsample2D


def test_downsample():
    checkpoint = 'stabilityai/stable-diffusion-2-1'

    unet = UNet2DConditionModel.from_pretrained(
        checkpoint, subfolder="unet",
    )

    for mod in unet.named_modules():
        if isinstance(mod[1], GoldDownsample2D):
            gold_block = mod[1]
            block = Downsample2D(
                channels=gold_block.channels,
                out_channels=gold_block.out_channels,
                use_conv=gold_block.use_conv,
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


if __name__ == '__main__':
    test_downsample()
