import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock as TransformerBlock  # noqa

from sd.transformer import BasicTransformerBlock


def test_transformer():
    checkpoint = 'stabilityai/stable-diffusion-2-1'

    unet = UNet2DConditionModel.from_pretrained(
        checkpoint, subfolder="unet",
    )

    for mod in unet.named_modules():
        if isinstance(mod[1], TransformerBlock):
            gold_block = mod[1]
            dim = gold_block.norm1.normalized_shape[0]
            num_attention_heads = gold_block.attn1.heads
            attention_head_dim = dim // num_attention_heads
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )

            block.load_state_dict(gold_block.state_dict())

            h = 32*64//(dim//10)
            hidden_states = torch.randn(40, h*h, dim)
            encoder_hidden_states = torch.randn(40, 77, 1024)

            gold = gold_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states)
            output = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states)
            delta = (gold - output).abs().max().item()
            print(mod[0], delta)
            assert delta < 1e-6


if __name__ == '__main__':
    test_transformer()
