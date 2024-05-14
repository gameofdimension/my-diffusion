import unittest

import torch
from timm.models.vision_transformer import Attention as GoldAttention
from timm.models.vision_transformer import Mlp as GoldMlp

from dit.blocks import Attention, Block, Mlp
from dit.model import Head
from tests.dit.gold_models import DiT, DiT_models, DiTBlock, FinalLayer


class BlockTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        image_size = 512
        num_classes = 1000
        device = 'cpu'
        latent_size = image_size // 8
        model_id = 'DiT-XL/2'
        model = DiT_models[model_id](
            input_size=latent_size,
            num_classes=num_classes
        ).to(device)
        cls.model = model
        cls.device = device

    @classmethod
    def tearDownClass(cls):
        pass

    def test_mlp(self):
        device = self.device
        model: DiT = self.model

        for mod in model.named_modules():
            if isinstance(mod[1], GoldMlp):
                gold_mod = mod[1]
                mod = Mlp(
                    in_features=gold_mod.fc1.in_features,
                    hidden_features=gold_mod.fc1.out_features)
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 8, gold_mod.fc1.in_features, device=device)
                out = mod(data)
                gold = gold_mod(data)
                delta = (gold-out).abs().max()
                assert delta < 1e-6

    def test_attention(self):
        device = self.device
        model: DiT = self.model

        for m in model.named_modules():
            if isinstance(m[1], GoldAttention):
                gold_mod = m[1]
                mod = Attention(dim=1152, num_heads=16)
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 8, 1152, device=device)
                out = mod(data)
                gold = gold_mod(data)
                delta = (gold-out).abs().max()
                assert delta < 1e-6

    def test_layer(self):
        device = self.device
        model: DiT = self.model

        for m in model.named_modules():
            if isinstance(m[1], DiTBlock):
                gold_mod = m[1]
                mod = Block(hidden_size=1152, num_heads=16)
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 8, 1152, device=device)
                condition = torch.randn(
                    10, 1152, device=device)

                out = mod(data, condition)
                gold = gold_mod(data, condition)
                delta = (gold-out).abs().max()
                assert delta < 1e-6

    def test_head(self):
        device = self.device
        model: DiT = self.model

        for m in model.named_modules():
            if isinstance(m[1], FinalLayer):
                gold_mod = m[1]
                mod = Head(hidden_size=1152, patch_size=2, out_channels=8)
                mod.load_state_dict(gold_mod.state_dict())

                data = torch.randn(
                    10, 8, 1152, device=device)
                condition = torch.randn(
                    10, 1152, device=device)

                out = mod(data, condition)
                gold = gold_mod(data, condition)
                delta = (gold-out).abs().max()
                assert delta < 1e-6
