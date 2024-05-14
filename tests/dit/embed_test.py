import unittest

import numpy as np
import torch

from dit.embedding import (LabelEmbed, PatchEmbed, TimestepEmbed,
                           get_2d_pos_embed)
from tests.dit.gold_models import DiT, DiT_models, get_2d_sincos_pos_embed


class EmbedTest(unittest.TestCase):

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

    def test_patch_embed(self):
        device = self.device
        model: DiT = self.model
        gold_mod = model.x_embedder

        hidden_size = 1152
        mod = PatchEmbed(
            img_size=64,
            patch_size=2,
            in_chans=4,
            embed_dim=hidden_size,
            bias=True
        )

        mod.load_state_dict(gold_mod.state_dict())
        data = torch.randn(10, 4, 64, 64, device=device)

        out = mod(data)
        gold = gold_mod(data)
        delta = (gold-out).abs().max()
        assert delta < 1e-6

    def test_time_embed(self):
        device = self.device
        model: DiT = self.model
        gold_mod = model.t_embedder

        hidden_size = 1152
        mod = TimestepEmbed(hidden_size)
        mod.load_state_dict(gold_mod.state_dict())

        t = torch.randint(0, 1000, (10,), device=device)
        out = mod(t)
        gold = gold_mod(t)
        delta = (gold-out).abs().max()
        assert delta < 1e-6

    def test_label_embed(self):
        device = self.device
        model: DiT = self.model
        gold_mod = model.y_embedder

        hidden_size = 1152
        mod = LabelEmbed(1000, hidden_size, 0.1)
        mod.load_state_dict(gold_mod.state_dict())

        mod.eval()
        gold_mod.eval()

        t = torch.randint(0, 1000+1, (10,), device=device)
        out = mod(t)
        gold = gold_mod(t, train=gold_mod.training)
        delta = (gold-out).abs().max()
        assert delta < 1e-6

    def test_pos_embed(self):
        embed_dim = 1152
        grid_size = 32
        gold = get_2d_sincos_pos_embed(
            embed_dim=embed_dim, grid_size=grid_size)
        out = get_2d_pos_embed(embed_dim, grid_size)

        delta = np.abs(gold-out).max()
        assert delta < 1e-6
