import unittest

import torch
from diffusers import UNet2DConditionModel

from sd.unet15 import CondtionalUNet15


class ModelTest(unittest.TestCase):

    def test_upsample(self):
        checkpoint = 'runwayml/stable-diffusion-v1-5'

        device = 'cuda'
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint, subfolder="unet",
        ).to(device)

        myunet = CondtionalUNet15().to(device)
        myunet.load_state_dict(unet.state_dict())

        bsz = 4
        for h, w in [(64, 64), (32, 32), (64, 48), (128, 96)]:
            latents = torch.randn(bsz, 4, h, w, device=device)
            timestep = torch.randint(0, 1000, (bsz, ), device=device)
            condition = torch.randn(bsz, 77, 768, device=device)

            out = myunet(latents, timestep, condition)[0]
            gold = unet(latents, timestep, condition, return_dict=False)[0]
            delta = (gold-out).abs().max().item()
            print(delta)
            assert delta < 1e-6
