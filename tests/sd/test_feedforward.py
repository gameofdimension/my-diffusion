import unittest

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import FeedForward as GoldFeedForward

from sd.transformer import FeedForward


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

    def test_ff(self):
        for mod in self.unet.named_modules():
            if isinstance(mod[1], GoldFeedForward):
                gold_ff = mod[1]
                dim = gold_ff.net[0].proj.weight.size(1)
                ff = FeedForward(
                    dim=dim,
                )
                ff.load_state_dict(gold_ff.state_dict())

                data = torch.randn(120, dim)

                output = ff(data)
                gold = gold_ff(data)

                delta = (gold - output).abs().max().item()
                assert delta < 1e-6
