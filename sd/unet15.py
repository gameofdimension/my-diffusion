import math
# import json
from typing import Tuple

import diffusers
import torch
# from deepspeed.accelerator import get_accelerator
# from deepspeed.profiling.flops_profiler import get_model_profile
from diffusers import UNet2DConditionModel
from transformers import PretrainedConfig


class CrossAttnDownBlock2D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


class DownBlock2D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


class UpBlock2D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


class CrossAttnUpBlock2D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass



class CondtionalUNet(torch.nn.Module):
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3

    def __init__(self) -> None:
        super().__init__()

        conv_in_padding = (self.conv_in_kernel - 1) // 2
        self.conv_in = torch.nn.Conv2d(
            self.in_channels, self.block_out_channels[0],
            kernel_size=self.conv_in_kernel, padding=conv_in_padding
        )

        timestep_input_dim = self.block_out_channels[0]
        time_embed_dim = timestep_input_dim * 4

        self.time_proj = Timesteps(
            timestep_input_dim, flip_sin_to_cos, freq_shift)

        self.down_blocks = torch.nn.ModuleList()
        self.middle_block = torch.nn.Module()
        self.up_blocks = torch.nn.ModuleList()

    def forward(self):
        pass


def main():
    pass


if __name__ == "__main__":
    # main()
    # checkpoint = 'runwayml/stable-diffusion-v1-5'
    checkpoint = 'stabilityai/stable-diffusion-2-1'

    config = PretrainedConfig.get_config_dict(checkpoint, subfolder='unet')[0]
    # print(json.dumps(config, indent=4))
    # cfg = AutoConfig.from_pretrained(checkpoint, subfolder='unet')
    # print(cfg)
    print(diffusers.utils.constants.USE_PEFT_BACKEND)

    device = 'cuda'
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint, subfolder="unet",
    ).to(device)

    # for m in unet.modules():
    #     print(m)

    bsz = 4
    latents = torch.randn(bsz, 4, 64, 64, device=device)
    timestep = torch.randint(0, 1000, (bsz, ), device=device)
    # condition = torch.randn(bsz, 77, 768, device=device)
    condition = torch.randn(bsz, 77, 1024, device=device)

    # with get_accelerator().device(0):
    #     flops, macs, params = get_model_profile(
    #         unet,
    #         args=[latents, timestep, condition],
    #         kwargs={"return_dict": False},
    #         print_profile=True,
    #         detailed=True,
    #         output_file='unet21_flops.txt',
    #     )

    #     out = unet(latents, timestep, condition, return_dict=False)
    # print(out[0].size())

    # att = unet.attn_processors
    # for k in att:
    #     print(k, att[k].__class__.__name__)

    # print(unet)
    for param in unet.named_parameters():
        print(param[0], param[1].size())
        break
