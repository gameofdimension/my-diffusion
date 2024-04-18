from typing import Any, Dict, Optional

import torch
from torch import nn


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(gate)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        bias: bool = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        assert activation_fn == "geglu"
        act_fn = GEGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

    def forward(
            self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
    ) -> None:
        super().__init__()

        self.inner_dim = dim_head * heads
        self.cross_attention_dim = cross_attention_dim or query_dim
        self.out_dim = query_dim
        self.heads = heads

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim,
                              self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim,
                              self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([])
        self.to_out.append(
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

    def forward(
            self, hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None) -> torch.Tensor:
        assert hidden_states.ndim == 3
        batch_size = hidden_states.shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(
            batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(
            batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for self.scale when we move to Torch 2.1
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'layer_norm_i2vgen'
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        assert norm_type == 'layer_norm'
        assert positional_embeddings is None
        assert not double_self_attention
        assert not only_cross_attention
        assert attention_type == 'default'

        super().__init__()
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        # self.use_ada_layer_norm_zero = (
        #     num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        # self.use_ada_layer_norm = (
        #     num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        # self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        # self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        # if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
        #     raise ValueError(
        #         f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
        #         f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
        #     )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        # if positional_embeddings and (num_positional_embeddings is None):
        #     raise ValueError(
        #         "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
        #     )

        # if positional_embeddings == "sinusoidal":
        #     self.pos_embed = SinusoidalPositionalEmbedding(
        #         dim, max_seq_length=num_positional_embeddings)
        # else:
        #     self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        # if norm_type == "ada_norm":
        #     self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        # elif norm_type == "ada_norm_zero":
        #     self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        # elif norm_type == "ada_norm_continuous":
        #     self.norm1 = AdaLayerNormContinuous(
        #         dim,
        #         ada_norm_continous_conditioning_embedding_dim,
        #         norm_elementwise_affine,
        #         norm_eps,
        #         ada_norm_bias,
        #         "rms_norm",
        #     )
        # else:
        self.norm1 = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        # if cross_attention_dim is not None or double_self_attention:
        #     We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
        #     I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
        #     the second cross attention block.
        #     if norm_type == "ada_norm":
        #         self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        #     elif norm_type == "ada_norm_continuous":
        #         self.norm2 = AdaLayerNormContinuous(
        #             dim,
        #             ada_norm_continous_conditioning_embedding_dim,
        #             norm_elementwise_affine,
        #             norm_eps,
        #             ada_norm_bias,
        #             "rms_norm",
        #         )
        #     else:
        self.norm2 = nn.LayerNorm(
            dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )  # is self-attn if encoder_hidden_states is none
        # else:
        #     self.norm2 = None
        #     self.attn2 = None

        # 3. Feed-forward
        # if norm_type == "ada_norm_continuous":
        #     self.norm3 = AdaLayerNormContinuous(
        #         dim,
        #         ada_norm_continous_conditioning_embedding_dim,
        #         norm_elementwise_affine,
        #         norm_eps,
        #         ada_norm_bias,
        #         "layer_norm",
        #     )

        # elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        # elif norm_type == "layer_norm_i2vgen":
        #     self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        # if attention_type == "gated" or attention_type == "gated-text-image":
        #     self.fuser = GatedSelfAttentionDense(
        #         dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        # if norm_type == "ada_norm_single":
        #     self.scale_shift_table = nn.Parameter(
        #         torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        assert attention_mask is None
        assert encoder_attention_mask is None
        assert timestep is None
        assert cross_attention_kwargs is None
        assert class_labels is None
        assert added_cond_kwargs is None
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # if self.norm_type == "ada_norm":
        #     norm_hidden_states = self.norm1(hidden_states, timestep)
        # elif self.norm_type == "ada_norm_zero":
        #     norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        #         hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        #     )
        # elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
        # elif self.norm_type == "ada_norm_continuous":
        #     norm_hidden_states = self.norm1(
        #         hidden_states, added_cond_kwargs["pooled_text_emb"])
        # elif self.norm_type == "ada_norm_single":
        #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #         self.scale_shift_table[None] +
        #         timestep.reshape(batch_size, 6, -1)
        #     ).chunk(6, dim=1)
        #     norm_hidden_states = self.norm1(hidden_states)
        #     norm_hidden_states = norm_hidden_states * \
        #         (1 + scale_msa) + shift_msa
        #     norm_hidden_states = norm_hidden_states.squeeze(1)
        # else:
        #     raise ValueError("Incorrect norm used")

        # if self.pos_embed is not None:
        #     norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get(
            "scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy(
        ) if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        # if self.norm_type == "ada_norm_zero":
        #     attn_output = gate_msa.unsqueeze(1) * attn_output
        # elif self.norm_type == "ada_norm_single":
        #     attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        # if gligen_kwargs is not None:
        #     hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        # if self.attn2 is not None:
        # if self.norm_type == "ada_norm":
        #     norm_hidden_states = self.norm2(hidden_states, timestep)
        # elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm2(hidden_states)
        # elif self.norm_type == "ada_norm_single":
        #     # For PixArt norm2 isn't applied here:
        #     # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
        #     norm_hidden_states = hidden_states
        # elif self.norm_type == "ada_norm_continuous":
        #     norm_hidden_states = self.norm2(
        #         hidden_states, added_cond_kwargs["pooled_text_emb"])
        # else:
        #     raise ValueError("Incorrect norm")

        # if self.pos_embed is not None and self.norm_type != "ada_norm_single":
        #     norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        # if self.norm_type == "ada_norm_continuous":
        #     norm_hidden_states = self.norm3(
        #         hidden_states, added_cond_kwargs["pooled_text_emb"])
        # elif not self.norm_type == "ada_norm_single":
        #     norm_hidden_states = self.norm3(hidden_states)

        # if self.norm_type == "ada_norm_zero":
        #     norm_hidden_states = norm_hidden_states * \
        #         (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # if self.norm_type == "ada_norm_single":
        #     norm_hidden_states = self.norm2(hidden_states)
        #     norm_hidden_states = norm_hidden_states * \
        #         (1 + scale_mlp) + shift_mlp

        # if self._chunk_size is not None:
        #     # "feed_forward_chunk_size" can be used to save memory
        #     ff_output = _chunked_feed_forward(
        #         self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
        #     )
        # else:
        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        # if self.norm_type == "ada_norm_zero":
        #     ff_output = gate_mlp.unsqueeze(1) * ff_output
        # elif self.norm_type == "ada_norm_single":
        #     ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Transformer2DModel(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale: float = None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        conv_cls = nn.Conv2d  # if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear  # if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (
            in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        # if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
        #     deprecation_message = (
        #         f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
        #         " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
        #         " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
        #         " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
        #         " would be very nice if you could open a Pull request for the `transformer/config.json` file"
        #     )
        #     deprecate("norm_type!=num_embeds_ada_norm", "1.0.0",
        #               deprecation_message, standard_warn=False)
        #     norm_type = "ada_norm"

        # if self.is_input_continuous and self.is_input_vectorized:
        #     raise ValueError(
        #         f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
        #         " sure that either `in_channels` or `num_vector_embeds` is None."
        #     )
        # elif self.is_input_vectorized and self.is_input_patches:
        #     raise ValueError(
        #         f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
        #         " sure that either `num_vector_embeds` or `num_patches` is None."
        #     )
        # elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
        #     raise ValueError(
        #         f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
        #         f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
        #     )

        # 2. Define input layers
        # if self.is_input_continuous:
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = linear_cls(in_channels, inner_dim)
        else:
            self.proj_in = conv_cls(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        # elif self.is_input_vectorized:
        #     assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
        #     assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

        #     self.height = sample_size
        #     self.width = sample_size
        #     self.num_vector_embeds = num_vector_embeds
        #     self.num_latent_pixels = self.height * self.width

        #     self.latent_image_embedding = ImagePositionalEmbeddings(
        #         num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
        #     )
        # elif self.is_input_patches:
        #     assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

        #     self.height = sample_size
        #     self.width = sample_size

        #     self.patch_size = patch_size
        #     interpolation_scale = (
        #         interpolation_scale if interpolation_scale is not None else max(
        #             self.config.sample_size // 64, 1)
        #     )
        #     self.pos_embed = PatchEmbed(
        #         height=sample_size,
        #         width=sample_size,
        #         patch_size=patch_size,
        #         in_channels=in_channels,
        #         embed_dim=inner_dim,
        #         interpolation_scale=interpolation_scale,
        #     )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # if self.is_input_continuous:
        # TODO: should use out_channels for continuous projections
        if use_linear_projection:
            self.proj_out = linear_cls(inner_dim, in_channels)
        else:
            self.proj_out = conv_cls(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        # elif self.is_input_vectorized:
        #     self.norm_out = nn.LayerNorm(inner_dim)
        #     self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        # elif self.is_input_patches and norm_type != "ada_norm_single":
        #     self.norm_out = nn.LayerNorm(
        #         inner_dim, elementwise_affine=False, eps=1e-6)
        #     self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
        #     self.proj_out_2 = nn.Linear(
        #         inner_dim, patch_size * patch_size * self.out_channels)
        # elif self.is_input_patches and norm_type == "ada_norm_single":
        #     self.norm_out = nn.LayerNorm(
        #         inner_dim, elementwise_affine=False, eps=1e-6)
        #     self.scale_shift_table = nn.Parameter(
        #         torch.randn(2, inner_dim) / inner_dim**0.5)
        #     self.proj_out = nn.Linear(
        #         inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        # if norm_type == "ada_norm_single":
        #     self.use_additional_conditions = self.config.sample_size == 128
        #     # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
        #     # additional conditions until we find better name
        #     self.adaln_single = AdaLayerNormSingle(
        #         inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        # if caption_channels is not None:
        #     self.caption_projection = PixArtAlphaTextProjection(
        #         in_features=caption_channels, hidden_size=inner_dim)

        # self.gradient_checkpointing = False

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if hasattr(module, "gradient_checkpointing"):
    #         module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        assert timestep is None
        assert added_cond_kwargs is None
        assert class_labels is None
        assert cross_attention_kwargs is None
        assert attention_mask is None
        assert encoder_attention_mask is None
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        # if attention_mask is not None and attention_mask.ndim == 2:
        #     # assume that mask is expressed as:
        #     #   (1 = keep,      0 = discard)
        #     # convert mask into a bias that can be added to attention scores:
        #     #       (keep = +0,     discard = -10000.0)
        #     attention_mask = (
        #         1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        #     attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        # if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        #     encoder_attention_mask = (
        #         1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        #     encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get(
            "scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        # if self.is_input_continuous:
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = (
                self.proj_in(hidden_states, scale=lora_scale)
                # if not USE_PEFT_BACKEND
                # else self.proj_in(hidden_states)
            )
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(
                0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(
                0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = (
                self.proj_in(hidden_states, scale=lora_scale)
                # if not USE_PEFT_BACKEND
                # else self.proj_in(hidden_states)
            )

        # elif self.is_input_vectorized:
        #     hidden_states = self.latent_image_embedding(hidden_states)
        # elif self.is_input_patches:
        #     height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        #     hidden_states = self.pos_embed(hidden_states)

        #     if self.adaln_single is not None:
        #         if self.use_additional_conditions and added_cond_kwargs is None:
        #             raise ValueError(
        #                 "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
        #             )
        #         batch_size = hidden_states.shape[0]
        #         timestep, embedded_timestep = self.adaln_single(
        #             timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        #         )

        # 2. Blocks
        # if self.caption_projection is not None:
        #     batch_size = hidden_states.shape[0]
        #     encoder_hidden_states = self.caption_projection(
        #         encoder_hidden_states)
        #     encoder_hidden_states = encoder_hidden_states.view(
        #         batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
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
            #         create_custom_forward(block),
            #         hidden_states,
            #         attention_mask,
            #         encoder_hidden_states,
            #         encoder_attention_mask,
            #         timestep,
            #         cross_attention_kwargs,
            #         class_labels,
            #         **ckpt_kwargs,
            #     )
            # else:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        # if self.is_input_continuous:
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(
                batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = (
                self.proj_out(hidden_states, scale=lora_scale)
                # if not USE_PEFT_BACKEND
                # else self.proj_out(hidden_states)
            )
        else:
            hidden_states = (
                self.proj_out(hidden_states, scale=lora_scale)
                # if not USE_PEFT_BACKEND
                # else self.proj_out(hidden_states)
            )
            hidden_states = hidden_states.reshape(
                batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        # elif self.is_input_vectorized:
        #     hidden_states = self.norm_out(hidden_states)
        #     logits = self.out(hidden_states)
        #     # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
        #     logits = logits.permute(0, 2, 1)

        #     # log(p(x_0))
        #     output = F.log_softmax(logits.double(), dim=1).float()

        # if self.is_input_patches:
        #     if self.config.norm_type != "ada_norm_single":
        #         conditioning = self.transformer_blocks[0].norm1.emb(
        #             timestep, class_labels, hidden_dtype=hidden_states.dtype
        #         )
        #         shift, scale = self.proj_out_1(
        #             F.silu(conditioning)).chunk(2, dim=1)
        #         hidden_states = self.norm_out(
        #             hidden_states) * (1 + scale[:, None]) + shift[:, None]
        #         hidden_states = self.proj_out_2(hidden_states)
        #     elif self.config.norm_type == "ada_norm_single":
        #         shift, scale = (
        #             self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        #         hidden_states = self.norm_out(hidden_states)
        #         # Modulation
        #         hidden_states = hidden_states * (1 + scale) + shift
        #         hidden_states = self.proj_out(hidden_states)
        #         hidden_states = hidden_states.squeeze(1)

        #     # unpatchify
        #     if self.adaln_single is None:
        #         height = width = int(hidden_states.shape[1] ** 0.5)
        #     hidden_states = hidden_states.reshape(
        #         shape=(-1, height, width, self.patch_size,
        #                self.patch_size, self.out_channels)
        #     )
        #     hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        #     output = hidden_states.reshape(
        #         shape=(-1, self.out_channels, height *
        #                self.patch_size, width * self.patch_size)
        #     )

        # if not return_dict:
        return (output,)

        # return Transformer2DModelOutput(sample=output)
