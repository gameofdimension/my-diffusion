import unittest

import torch

from tests.uvit.timm import Mlp as GoldMlp
from tests.uvit.uvit_t2i import Attention as GoldAttention
from tests.uvit.uvit_t2i import Block as GoldBlock
from tests.uvit.uvit_t2i import UViT
from uvit.blocks import Attention, Block, Mlp
from uvit.config import t2i_config


class BlocksTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        device = 'cpu'
        config = t2i_config()
        model = UViT(**config)

        path = 'pretrained_models/mscoco_uvit_small_deep.pth'
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        cls.model = model
        cls.device = device
        cls.config = config

    @classmethod
    def tearDownClass(cls):
        pass

    def test_mlp(self):
        device = self.device
        model: UViT = self.model

        for m in model.named_modules():
            if isinstance(m[1], GoldMlp):
                gold_mod = m[1]
                mod = Mlp(
                    in_features=gold_mod.fc1.in_features,
                    hidden_features=gold_mod.fc1.out_features,
                    out_features=gold_mod.fc2.out_features,
                )
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 8, gold_mod.fc1.in_features, device=device)
                out = mod(data)
                gold = gold_mod(data)
                delta = (gold-out).abs().max()
                assert delta < 1e-6

    def test_attention(self):
        device = self.device
        model: UViT = self.model

        for m in model.named_modules():
            if isinstance(m[1], GoldAttention):
                gold_mod = m[1]
                mod = Attention(
                    dim=self.config['embed_dim'],
                    num_heads=self.config['num_heads'],
                    qkv_bias=self.config['qkv_bias'],
                ).to(device=device)
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 334, self.config['embed_dim'], device=device)
                out = mod(data)
                gold = gold_mod(data)
                delta = (gold-out).abs().max()

                assert delta < 1e-6

    def test_block(self):
        device = self.device
        model: UViT = self.model

        for m in model.named_modules():
            if isinstance(m[1], GoldBlock):
                gold_mod = m[1]
                has_skip = gold_mod.skip_linear is not None
                mod = Block(
                    dim=self.config['embed_dim'],
                    num_heads=self.config['num_heads'],
                    skip=has_skip,
                ).to(device=device)
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 334, self.config['embed_dim'], device=device)
                skip = torch.rand_like(data) if has_skip else None
                out = mod(data, skip=skip)
                gold = gold_mod(data, skip=skip)
                delta = (gold-out).abs().max()

                assert delta < 1e-6
