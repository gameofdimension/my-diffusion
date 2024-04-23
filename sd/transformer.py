from typing import Optional

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
            self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
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
        cross_attention_dim: int,
    ):
        super().__init__()
        dropout = 0.0
        activation_fn: str = "geglu"
        attention_bias: bool = False
        only_cross_attention: bool = False
        norm_elementwise_affine: bool = True
        norm_type: str = "layer_norm"
        norm_eps: float = 1e-5
        ff_bias: bool = True
        attention_out_bias: bool = True

        assert norm_type == 'layer_norm'
        assert not only_cross_attention

        self.only_cross_attention = only_cross_attention
        self.use_layer_norm = norm_type == "layer_norm"
        self.norm_type = norm_type

        self.norm1 = nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
        )

        self.norm2 = nn.LayerNorm(
            dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
        )
        hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        return hidden_states


class Transformer2DModel(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        cross_attention_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        norm_num_groups: int = 32
        use_linear_projection: bool = True
        out_channels = in_channels

        assert norm_num_groups == 32
        assert use_linear_projection
        self.use_linear_projection = use_linear_projection
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        self.out_channels = out_channels
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(
            0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        return (output,)
