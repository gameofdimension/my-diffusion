from typing import Tuple

import torch

from sd.blocks import (CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D,
                       UNetMidBlock2DCrossAttn, UpBlock2D)
from sd.time_embed import TimestepEmbedding, Timesteps


class CondtionalUNet15(torch.nn.Module):
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
    num_attention_heads = 8
    num_upsamplers = 3
    cross_attention_dim = 768
    use_linear_projection = False
    layers_per_block: int = 2

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

        self.down_blocks = torch.nn.ModuleList([
            CrossAttnDownBlock2D(
                in_channels=self.block_out_channels[0],
                out_channels=self.block_out_channels[0],
                num_attention_heads=self.num_attention_heads,
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block,
                transformer_layers_per_block=[1, 1],
                cross_attention_dim=self.cross_attention_dim,
                use_linear_projection=self.use_linear_projection,
                add_downsample=True,
            ),
            CrossAttnDownBlock2D(
                in_channels=self.block_out_channels[0],
                out_channels=self.block_out_channels[1],
                num_attention_heads=self.num_attention_heads,
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block,
                transformer_layers_per_block=[1, 1],
                cross_attention_dim=self.cross_attention_dim,
                use_linear_projection=self.use_linear_projection,
                add_downsample=True,

            ),
            CrossAttnDownBlock2D(
                in_channels=self.block_out_channels[1],
                out_channels=self.block_out_channels[2],
                num_attention_heads=self.num_attention_heads,
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block,
                transformer_layers_per_block=[1, 1],
                cross_attention_dim=self.cross_attention_dim,
                use_linear_projection=self.use_linear_projection,
                add_downsample=True,
            ),
            DownBlock2D(
                in_channels=self.block_out_channels[2],
                out_channels=self.block_out_channels[3],
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block,
                add_downsample=False,
            ),
        ])
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=self.block_out_channels[-1],
            temb_channels=time_embed_dim,
            num_attention_heads=self.num_attention_heads,
            num_layers=1,
            transformer_layers_per_block=[1],
            cross_attention_dim=self.cross_attention_dim,
            use_linear_projection=self.use_linear_projection,
        )
        self.up_blocks = torch.nn.ModuleList([
            UpBlock2D(
                in_channels=self.block_out_channels[-2],
                out_channels=self.block_out_channels[-1],
                prev_output_channel=self.block_out_channels[-1],
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block+1,
                add_upsample=True,
            ),
            CrossAttnUpBlock2D(
                in_channels=self.block_out_channels[-3],
                out_channels=self.block_out_channels[-2],
                prev_output_channel=self.block_out_channels[-1],
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block+1,
                transformer_layers_per_block=[1, 1, 1],
                num_attention_heads=self.num_attention_heads,
                cross_attention_dim=self.cross_attention_dim,
                use_linear_projection=self.use_linear_projection,
                add_upsample=True,
            ),
            CrossAttnUpBlock2D(
                in_channels=self.block_out_channels[-4],
                out_channels=self.block_out_channels[-3],
                prev_output_channel=self.block_out_channels[-2],
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block+1,
                transformer_layers_per_block=[1, 1, 1],
                num_attention_heads=self.num_attention_heads,
                cross_attention_dim=self.cross_attention_dim,
                use_linear_projection=self.use_linear_projection,
                add_upsample=True,
            ),
            CrossAttnUpBlock2D(
                in_channels=self.block_out_channels[-4],
                out_channels=self.block_out_channels[-4],
                prev_output_channel=self.block_out_channels[-3],
                temb_channels=time_embed_dim,
                num_layers=self.layers_per_block+1,
                transformer_layers_per_block=[1, 1, 1],
                num_attention_heads=self.num_attention_heads,
                cross_attention_dim=self.cross_attention_dim,
                use_linear_projection=self.use_linear_projection,
                add_upsample=False,
            ),
        ])

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
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        assert self.mid_block is not None
        if self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample, emb,
                encoder_hidden_states=encoder_hidden_states)
        else:
            sample = self.mid_block(sample, emb)

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
