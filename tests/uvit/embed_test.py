import unittest

import torch

from tests.uvit.uvit_t2i import UViT, timestep_embedding
from uvit.config import t2i_config
from uvit.embedding import PatchEmbed, TimestepEmbed


class EmbedTest(unittest.TestCase):

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

    def test_patch_embed(self):
        model = self.model
        device = self.device
        gold_mod = model.patch_embed
        mod = PatchEmbed(
            patch_size=self.config['patch_size'],
            in_chans=self.config['in_chans'],
            embed_dim=self.config['embed_dim'],
        ).to(device=device)
        mod.load_state_dict(gold_mod.state_dict())

        x = torch.randn(
            10,
            self.config['in_chans'],
            self.config['img_size'],
            self.config['img_size'],
        ).to(device)
        out = mod(x)
        gold = gold_mod(x)
        delta = (gold - out).abs().max()
        assert delta < 1e-6

    def test_time_embed(self):
        model = self.model
        device = self.device

        gold_mod = model.time_embed
        mod = TimestepEmbed(
            dim=self.config['embed_dim'],
        ).to(device=device)

        t = torch.randint(0, 1000, (101,)).to(device)
        out = mod(t)
        gold = gold_mod((timestep_embedding(t, self.config['embed_dim'])))
        delta = (gold - out).abs().max()
        assert delta < 1e-6
