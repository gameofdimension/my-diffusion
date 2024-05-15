import unittest

import torch

from tests.uvit.uvit_t2i import UViT
from uvit.config import t2i_config
from uvit.model import Model


class ModelTest(unittest.TestCase):

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

    def test_model(self):
        gold_mod = self.model
        mod = Model(
            img_size=self.config['img_size'],
            patch_size=self.config['patch_size'],
            in_chans=self.config['in_chans'],
            embed_dim=self.config['embed_dim'],
            depth=self.config['depth'],
            num_heads=self.config['num_heads'],
            clip_dim=self.config['clip_dim'],
            num_clip_token=self.config['num_clip_token'],
        ).to(device=self.device)
        mod.load_state_dict(gold_mod.state_dict())

        x = torch.randn(
            10,
            self.config['in_chans'],
            self.config['img_size'],
            self.config['img_size'],
        ).to(device=self.device)
        t = torch.randint(0, 1000, (10,)).to(device=self.device)
        c = torch.randn(
            10, self.config['num_clip_token'], self.config['clip_dim']
        ).to(device=self.device)
        out = mod(x, t, c)
        gold = gold_mod(x, t, c)
        delta = (gold-out).abs().max()
        assert delta < 1e-6
