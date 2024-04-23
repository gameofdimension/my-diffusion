import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import \
    BasicTransformerBlock as TransformerBlock
from diffusers.models.transformers.transformer_2d import \
    Transformer2DModel as GoldTransformer

from sd.transformer import BasicTransformerBlock, Transformer2DModel


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

    def test_block(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], TransformerBlock):
                gold_block = mod[1]
                dim = gold_block.norm1.normalized_shape[0]
                num_attention_heads = gold_block.attn1.heads
                attention_head_dim = dim // num_attention_heads
                block = BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=1024,
                )

                block.load_state_dict(gold_block.state_dict())

                h = 32*64//(dim//10)
                hidden_states = torch.randn(40, h*h, dim)
                encoder_hidden_states = torch.randn(40, 77, 1024)

                gold = gold_block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states)
                output = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states)
                delta = (gold - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6

    def test_transformer(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldTransformer):
                attention_head_dim = 64
                gold_transformer = mod[1]
                num_attention_heads = gold_transformer.in_channels // attention_head_dim  # noqa
                transformer = Transformer2DModel(
                    in_channels=gold_transformer.in_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=gold_transformer.in_channels // num_attention_heads,  # noqa
                    cross_attention_dim=1024,
                    num_layers=1,
                )

                h = 32*64//(gold_transformer.in_channels//10)
                hidden_states = torch.randn(
                    40, gold_transformer.in_channels, h, h)
                encoder_hidden_states = torch.randn(40, 77, 1024)
                transformer.load_state_dict(gold_transformer.state_dict())

                output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states)[0]

                gold_output = gold_transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states)[0]
                delta = (gold_output - output).abs().max().item()
                print(mod[0], delta)
                assert delta < 1e-6
