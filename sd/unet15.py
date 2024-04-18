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


class CrossAttnDownBlock2D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [
                transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            # if not dual_cross_attention:
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )
            # else:
            #     attentions.append(
            #         DualTransformer2DModel(
            #             num_attention_heads,
            #             out_channels // num_attention_heads,
            #             in_channels=out_channels,
            #             num_layers=1,
            #             cross_attention_dim=cross_attention_dim,
            #             norm_num_groups=resnet_groups,
            #         )
            #     )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self):
        output_states = ()

        lora_scale = cross_attention_kwargs.get(
            "scale", 1.0) if cross_attention_kwargs is not None else 1.0

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            # if self.training and self.gradient_checkpointing:

            #     def create_custom_forward(module, return_dict=None):
            #         def custom_forward(*inputs):
            #             if return_dict is not None:
            #                 return module(*inputs, return_dict=return_dict)
            #             else:
            #                 return module(*inputs)

            #         return custom_forward

            #     ckpt_kwargs: Dict[str, Any] = {
            #         "use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            #     hidden_states = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(resnet),
            #         hidden_states,
            #         temb,
            #         **ckpt_kwargs,
            #     )
            #     hidden_states = attn(
            #         hidden_states,
            #         encoder_hidden_states=encoder_hidden_states,
            #         cross_attention_kwargs=cross_attention_kwargs,
            #         attention_mask=attention_mask,
            #         encoder_attention_mask=encoder_attention_mask,
            #         return_dict=False,
            #     )[0]
            # else:
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


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

    for mod in unet.named_modules():
        print(mod[0], mod[1].__class__.__name__)