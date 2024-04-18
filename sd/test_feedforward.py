import torch
from diffusers import UNet2DConditionModel

from sd.transformer import FeedForward
from diffusers.models.attention import FeedForward as GoldFeedForward


def test_ff():
    checkpoint = 'stabilityai/stable-diffusion-2-1'

    unet = UNet2DConditionModel.from_pretrained(
        checkpoint, subfolder="unet",
    )

    for mod in unet.named_modules():
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
            print(mod[0], (gold - output).abs().max().item())


if __name__ == '__main__':
    test_ff()
