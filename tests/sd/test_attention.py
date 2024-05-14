import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import Attention as GoldAttention

from sd.transformer import Attention


class Unet21ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        checkpoint = 'stabilityai/stable-diffusion-2-1'
        cls.unet = UNet2DConditionModel.from_pretrained(
            checkpoint, subfolder="unet",
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def test_attention(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldAttention):
                gold_attn = mod[1]
                query_dim = gold_attn.query_dim
                cross_attention_dim = None
                if mod[0].endswith('attn1'):
                    attn = Attention(
                        query_dim=query_dim,
                        cross_attention_dim=None,
                        heads=gold_attn.heads,
                        dim_head=64,
                    )
                else:
                    cross_attention_dim = gold_attn.cross_attention_dim
                    attn = Attention(
                        query_dim=query_dim,
                        cross_attention_dim=cross_attention_dim,
                        heads=gold_attn.heads,
                        dim_head=64,
                    )

                attn.load_state_dict(gold_attn.state_dict())
                data = torch.randn(40, 120, query_dim)

                if cross_attention_dim is None:
                    output = attn(data)
                    gold = gold_attn(data)
                else:
                    cross_data = torch.randn(40, 210, cross_attention_dim)
                    output = attn(data, cross_data)
                    gold = gold_attn(data, cross_data)

                delta = (gold - output).abs().max().item()
                assert delta < 1e-6
