from diffusers import UNet2DConditionModel
from transformers import PretrainedConfig

import json
import diffusers
import torch


def main():
    pass


if __name__ == "__main__":
    # main()
    checkpoint = 'runwayml/stable-diffusion-v1-5'

    config = PretrainedConfig.get_config_dict(checkpoint, subfolder='unet')[0]
    print(json.dumps(config, indent=4))
    # cfg = AutoConfig.from_pretrained(checkpoint, subfolder='unet')
    # print(cfg)
    print(diffusers.utils.constants.USE_PEFT_BACKEND)

    device = 'cuda'
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint, subfolder="unet",
    ).to(device)

    bsz = 4
    latents = torch.randn(bsz, 4, 64, 64, device=device)
    timestep = torch.randint(0, 1000, (bsz, ), device=device)
    condition = torch.randn(bsz, 77, 768, device=device)

    out = unet(latents, timestep, condition, return_dict=False)
    print(out[0].size())

    # att = unet.attn_processors
    # for k in att:
    #     print(k, att[k].__class__.__name__)

    # # print(unet)
    # for param in unet.named_parameters():
    #     print(param[0], param[1].size())
