import math
# import json
from typing import Tuple

import diffusers
import torch
# from deepspeed.accelerator import get_accelerator
# from deepspeed.profiling.flops_profiler import get_model_profile
from diffusers import UNet2DConditionModel
from transformers import PretrainedConfig

from sd.time_embed import TimestepEmbedding, Timesteps


class CondtionalUNet(torch.nn.Module):
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3
    flip_sin_to_cos: bool = True
    freq_shift = 0
    act_fn = 'silu'
    norm_num_groups: int = 32
    norm_eps: float = 1e-5

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
            timestep_input_dim, self.flip_sin_to_cos, self.freq_shift)
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim)

        self.down_blocks = torch.nn.ModuleList([])
        self.middle_block = torch.nn.Module([])
        self.up_blocks = torch.nn.ModuleList([])

        self.conv_norm_out = torch.nn.GroupNorm(
            num_channels=self.block_out_channels[0],
            num_groups=self.norm_num_groups, eps=self.norm_eps
        )
        self.conv_act = torch.nn.SiLU()

        conv_out_padding = (self.conv_out_kernel - 1) // 2
        self.conv_out = torch.nn.Conv2d(
            self.block_out_channels[0], self.out_channels,
            kernel_size=self.conv_out_kernel, padding=conv_out_padding
        )

    def embedding_time(self, dtype, timestep):
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=dtype)
        emb = self.time_embedding(t_emb)
        return emb

    def forward(self, sample, timestep, encoder_hidden_states):
        default_overall_up_factor = 2**self.num_upsamplers
        assert sample.size(-1) % default_overall_up_factor == 0
        assert sample.size(-2) % default_overall_up_factor == 0

        emb = self.embedding_time(sample.dtype, timestep)
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        assert self.middle_block is not None
        if self.middle_block.has_cross_attention:
            sample = self.middle_block(
                sample, emb,
                encoder_hidden_states=encoder_hidden_states)
        else:
            sample = self.middle_block(sample, emb)

        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]

            if upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return (sample,)


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

    out = unet(latents, timestep, condition, return_dict=False)
    # print(out[0].size())

    # att = unet.attn_processors
    # for k in att:
    #     print(k, att[k].__class__.__name__)

    # print(unet)
    for param in unet.named_parameters():
        print(param[0], param[1].size())
        break

    # for mod in unet.named_modules():
    #     # print(mod[0], mod[1].__class__.__name__)
    #     if isinstance(mod[1], GoldAttention):
    #         for param in mod[1].named_parameters():
    #             print("------", mod[0], param[0], param[1].size())
