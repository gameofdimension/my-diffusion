from typing import Any, Dict, Tuple, Optional, Union
import torch
from torch import nn


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        # conv_cls = nn.Conv2d  # if USE_PEFT_BACKEND else LoRACompatibleConv

        # print("------------------------", norm_type, name)
        # if norm_type == "ln_norm":
        #     self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        # elif norm_type == "rms_norm":
        #     self.norm = RMSNorm(channels, eps, elementwise_affine)
        # elif norm_type is None:
        #     self.norm = None
        # else:
        #     raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        # if name == "conv":
        #     self.Conv2d_0 = conv
        #     self.conv = conv
        # elif name == "Conv2d_0":
        #     self.conv = conv
        # else:
        #     self.conv = conv
        self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        # if self.norm is not None:
        #     hidden_states = self.norm(
        #         hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = torch.nn.functional.pad(
                hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        # if not USE_PEFT_BACKEND:
        #     if isinstance(self.conv, LoRACompatibleConv):
        #         hidden_states = self.conv(hidden_states, scale)
        #     else:
        #         hidden_states = self.conv(hidden_states)
        # else:
        #     hidden_states = self.conv(hidden_states)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock2D(torch.nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift"
            for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[torch.FloatTensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        # if time_embedding_norm == "ada_group":
        #     raise ValueError(
        #         "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
        #     )
        # if time_embedding_norm == "spatial":
        #     raise ValueError(
        #         "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
        #     )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        # self.up = up
        # self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear  # if USE_PEFT_BACKEND else LoRACompatibleLinear
        conv_cls = nn.Conv2d  # if USE_PEFT_BACKEND else LoRACompatibleConv

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = conv_cls(in_channels, out_channels,
                              kernel_size=3, stride=1, padding=1)

        # if temb_channels is not None:
        #     if self.time_embedding_norm == "default":
        self.time_emb_proj = linear_cls(temb_channels, out_channels)
        #     elif self.time_embedding_norm == "scale_shift":
        #         self.time_emb_proj = linear_cls(
        #             temb_channels, 2 * out_channels)
        #     else:
        #         raise ValueError(
        #             f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        # else:
        #     self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = conv_cls(out_channels, conv_2d_out_channels,
                              kernel_size=3, stride=1, padding=1)

        self.nonlinearity = torch.nn.SiLU()  # get_activation(non_linearity)

        self.upsample = self.downsample = None
        # if self.up:
        #     if kernel == "fir":
        #         fir_kernel = (1, 3, 3, 1)
        #         self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
        #     elif kernel == "sde_vp":
        #         self.upsample = partial(
        #             F.interpolate, scale_factor=2.0, mode="nearest")
        #     else:
        #         self.upsample = Upsample2D(in_channels, use_conv=False)
        # elif self.down:
        #     if kernel == "fir":
        #         fir_kernel = (1, 3, 3, 1)
        #         self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
        #     elif kernel == "sde_vp":
        #         self.downsample = partial(
        #             F.avg_pool2d, kernel_size=2, stride=2)
        #     else:
        #         self.downsample = Downsample2D(
        #             in_channels, use_conv=False, padding=1, name="op")

        # if use_in_shortcut is None else use_in_shortcut
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        # if self.upsample is not None:
        #     # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        #     if hidden_states.shape[0] >= 64:
        #         input_tensor = input_tensor.contiguous()
        #         hidden_states = hidden_states.contiguous()
        #     input_tensor = (
        #         self.upsample(input_tensor, scale=scale)
        #         if isinstance(self.upsample, Upsample2D)
        #         else self.upsample(input_tensor)
        #     )
        #     hidden_states = (
        #         self.upsample(hidden_states, scale=scale)
        #         if isinstance(self.upsample, Upsample2D)
        #         else self.upsample(hidden_states)
        #     )
        # elif self.downsample is not None:
        #     input_tensor = (
        #         self.downsample(input_tensor, scale=scale)
        #         if isinstance(self.downsample, Downsample2D)
        #         else self.downsample(input_tensor)
        #     )
        #     hidden_states = (
        #         self.downsample(hidden_states, scale=scale)
        #         if isinstance(self.downsample, Downsample2D)
        #         else self.downsample(hidden_states)
        #     )

        hidden_states = self.conv1(
            hidden_states, scale)  # if not USE_PEFT_BACKEND else self.conv1(hidden_states)

        # if self.time_emb_proj is not None:
        #     if not self.skip_time_act:
        temb = self.nonlinearity(temb)
        temb = (
            self.time_emb_proj(temb, scale)[:, :, None, None]
            # if not USE_PEFT_BACKEND
            # else self.time_emb_proj(temb)[:, :, None, None]
        )

        # if self.time_embedding_norm == "default":
        #     if temb is not None:
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        # elif self.time_embedding_norm == "scale_shift":
        #     if temb is None:
        #         raise ValueError(
        #             f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
        #         )
        #     time_scale, time_shift = torch.chunk(temb, 2, dim=1)
        #     hidden_states = self.norm2(hidden_states)
        #     hidden_states = hidden_states * (1 + time_scale) + time_shift
        # else:
        #     hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(
            hidden_states, scale)  # if not USE_PEFT_BACKEND else self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(
                    input_tensor, scale)  # if not USE_PEFT_BACKEND else self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / \
            self.output_scale_factor

        return output_tensor


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

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

        # self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            # if self.training and self.gradient_checkpointing:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs)

            #         return custom_forward

            #     if is_torch_version(">=", "1.11.0"):
            #         hidden_states = torch.utils.checkpoint.checkpoint(
            #             create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
            #         )
            #     else:
            #         hidden_states = torch.utils.checkpoint.checkpoint(
            #             create_custom_forward(resnet), hidden_states, temb
            #         )
            # else:
            hidden_states = resnet(hidden_states, temb, scale=scale)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
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

        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        assert attention_mask is None
        assert cross_attention_kwargs is None
        assert encoder_attention_mask is None
        assert additional_residuals is None

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
