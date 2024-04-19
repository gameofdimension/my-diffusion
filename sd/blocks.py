from typing import Tuple, Optional, Union, Dict, Any

import torch
from torch import nn

from sd.transformer import Transformer2DModel


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        out_channels: int,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = torch.nn.functional.pad(
                hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        groups: int = 32,
    ):
        super().__init__()
        conv_shortcut_bias: bool = True
        output_scale_factor: float = 1.0
        time_embedding_norm: str = "default"
        eps: float = 1e-5
        conv_shortcut: bool = False
        skip_time_act: bool = False

        self.pre_norm = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act
        groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels,
            eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1)

        self.nonlinearity = torch.nn.SiLU()
        self.upsample = self.downsample = None
        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / \
            self.output_scale_factor

        return output_tensor


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
    ):
        super().__init__()
        dropout: float = 0.0
        resnet_groups: int = 32
        add_downsample: bool = False
        downsample_padding: int = 1

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self, hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        transformer_layers_per_block: Tuple[int],
        num_attention_heads: int = 1,
        add_downsample: bool = True,
    ):
        super().__init__()
        downsample_padding: int = 1
        resnets = []
        attentions = []

        self.num_attention_heads = num_attention_heads
        assert len(transformer_layers_per_block) == num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )[0]
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        name: str = "conv",
        padding=1,
        bias=True,
        interpolate=True,
    ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels
        self.name = name
        self.interpolate = interpolate
        kernel_size = 3
        conv = nn.Conv2d(
            self.channels, self.out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = torch.nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
    ):
        super().__init__()
        dropout: float = 0.0
        num_layers: int = 3
        resnet_groups: int = 32
        add_upsample: bool = True

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels  # noqa
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        out_channels=out_channels)
                ])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: torch.FloatTensor,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat(
                [hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        # resolution_idx: Optional[int] = None,
        # dropout: float = 0.0,
        num_layers: int,
        # transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        transformer_layers_per_block: Tuple[int],
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        # self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        assert len(transformer_layers_per_block) == num_layers

        # if isinstance(transformer_layers_per_block, int):
        #     transformer_layers_per_block = [
        #         transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    # eps=resnet_eps,
                    # groups=resnet_groups,
                    # dropout=dropout,
                    # time_embedding_norm=resnet_time_scale_shift,
                    # non_linearity=resnet_act_fn,
                    # output_scale_factor=output_scale_factor,
                    # pre_norm=resnet_pre_norm,
                )
            )
            # if not dual_cross_attention:
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    # out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    # cross_attention_dim=cross_attention_dim,
                    # norm_num_groups=resnet_groups,
                    # use_linear_projection=use_linear_projection,
                    # only_cross_attention=only_cross_attention,
                    # upcast_attention=upcast_attention,
                    # attention_type=attention_type,
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

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(
                    out_channels,
                    # use_conv=True,
                    out_channels=out_channels)])
        else:
            self.upsamplers = None

        # self.gradient_checkpointing = False
        # self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # upsample_size: Optional[int] = None,
        # attention_mask: Optional[torch.FloatTensor] = None,
        # encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # lora_scale = cross_attention_kwargs.get(
        #     "scale", 1.0) if cross_attention_kwargs is not None else 1.0
        # is_freeu_enabled = (
        #     getattr(self, "s1", None)
        #     and getattr(self, "s2", None)
        #     and getattr(self, "b1", None)
        #     and getattr(self, "b2", None)
        # )

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            # if is_freeu_enabled:
            #     hidden_states, res_hidden_states = apply_freeu(
            #         self.resolution_idx,
            #         hidden_states,
            #         res_hidden_states,
            #         s1=self.s1,
            #         s2=self.s2,
            #         b1=self.b1,
            #         b2=self.b2,
            #     )

            hidden_states = torch.cat(
                [hidden_states, res_hidden_states], dim=1)

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
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                # cross_attention_kwargs=cross_attention_kwargs,
                # attention_mask=attention_mask,
                # encoder_attention_mask=encoder_attention_mask,
                # return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(
                    hidden_states)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(
            in_channels // 4, 32)

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [
                transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for i in range(num_layers):
            # if not dual_cross_attention:
            attentions.append(
                Transformer2DModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )
            # else:
            #     attentions.append(
            #         DualTransformer2DModel(
            #             num_attention_heads,
            #             in_channels // num_attention_heads,
            #             in_channels=in_channels,
            #             num_layers=1,
            #             cross_attention_dim=cross_attention_dim,
            #             norm_num_groups=resnet_groups,
            #         )
            #     )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
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

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # lora_scale = cross_attention_kwargs.get(
        #     "scale", 1.0) if cross_attention_kwargs is not None else 1.0
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
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
            #     hidden_states = attn(
            #         hidden_states,
            #         encoder_hidden_states=encoder_hidden_states,
            #         cross_attention_kwargs=cross_attention_kwargs,
            #         attention_mask=attention_mask,
            #         encoder_attention_mask=encoder_attention_mask,
            #         return_dict=False,
            #     )[0]
            #     hidden_states = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(resnet),
            #         hidden_states,
            #         temb,
            #         **ckpt_kwargs,
            #     )
            # else:
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                # cross_attention_kwargs=cross_attention_kwargs,
                # attention_mask=attention_mask,
                # encoder_attention_mask=encoder_attention_mask,
                # return_dict=False,
            )[0]
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)

        return hidden_states
