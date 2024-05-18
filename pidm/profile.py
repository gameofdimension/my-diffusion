import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile
from torchprofile import profile_macs

from pidm.model import Model


def make_model(image_size):
    return Model(
        image_size=image_size,
        in_channels=23,
        out_channels=6,
        model_channels=128,
        embed_channels=512,
        channel_mult=[1, 1, 2, 2, 4, 4],
        attention_resolutions=[image_size//8, image_size//16, image_size//32],
        dropout=0.0,
    )


def deepspeed_profile(model, args):
    with get_accelerator().device(0), torch.no_grad():
        flops, macs, params = get_model_profile(
            model=model,
            args=args,
            kwargs=None,
            print_profile=False,
            # detailed=True,
            module_depth=-1,
            top_modules=1,
            warm_up=1,
            as_string=False,
            output_file=None,
            ignore_modules=None
        )
    return macs


def profile(prof_func):
    device = 'cuda'
    h, w = 256*2*2, 256*2*2
    model = make_model(h).to(device)
    bsz = 1

    noise = torch.randn(bsz, 23, h, w, device=device)
    timestep = torch.randint(0, 1000, (bsz, ), device=device)
    ref = torch.randn(bsz, 3, h, w, device=device)
    out = prof_func(
        model, args=(noise, timestep, ref))
    return f"{out/1e9:.3f} GMACs"


if __name__ == '__main__':
    print(profile(profile_macs))
    print(profile(deepspeed_profile))
