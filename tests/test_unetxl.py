import unittest

import torch
from diffusers import UNet2DConditionModel

from sd.unetxl import CondtionalUNetXL


class ModelTest(unittest.TestCase):

    def test_unet(self):
        checkpoint = 'stabilityai/stable-diffusion-xl-base-1.0'

        device = 'cuda'
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint, subfolder="unet",
        ).to(device)

        bsz = 4
        h, w = 64, 64
        latents = torch.randn(bsz, 4, h, w, device=device)
        timestep = torch.randint(0, 1000, (bsz, ), device=device)
        condition = torch.randn(bsz, 77, 2048, device=device)
        time_ids = torch.randn(bsz, 6, device=device)
        text_embeds = torch.randn(bsz, 1280, device=device)

        gold = unet(
            latents, timestep, condition,
            added_cond_kwargs={
                "time_ids": time_ids,
                "text_embeds": text_embeds
            },
            return_dict=False,
        )[0]

        myunet = CondtionalUNetXL().to(device)
        myunet.load_state_dict(unet.state_dict())
        output = myunet(
            latents, timestep, condition,
            time_ids, text_embeds
        )[0]

        delta = (gold-output).abs().max().item()
        assert delta < 1e-6
