import unittest

import torch

from dit.model import Model
from tests.dit.gold_models import DiT, DiT_models


class ModelTest(unittest.TestCase):

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
        ckpt_path = f"pretrained_models/DiT-XL-2-{image_size}x{image_size}.pt"
        state_dict = torch.load(ckpt_path, map_location=lambda s, loc: s)
        model.load_state_dict(state_dict)
        model.eval()  # important!
        cls.model = model
        cls.device = device

    @classmethod
    def tearDownClass(cls):
        pass

    def test_forward(self):
        device = self.device
        gold_mod: DiT = self.model
        mod = Model(
            input_size=64,
            hidden_size=1152,
            in_channels=4,
            patch_size=2,
            num_heads=16,
            num_layers=28,
            class_dropout_prob=0.1,
        )
        mod.load_state_dict(gold_mod.state_dict())
        mod.eval()

        x = torch.randn(10, 4, 64, 64, device=device)
        y = torch.randint(0, 1000+1, (10,), device=device)
        t = torch.randint(0, 1000, (10,), device=device)

        out = mod(x=x, t=t, y=y)
        gold = gold_mod(x=x, t=t, y=y)
        delta = (gold-out).abs().max()
        assert delta < 1e-6
