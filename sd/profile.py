import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from torchprofile import profile_macs

from sd.unet15 import CondtionalUNet15
from sd.unet21 import CondtionalUNet21
from sd.unetxl import CondtionalUNetXL


def make_model(type: str):
    if type == 'sd15':
        return CondtionalUNet15()
    if type == 'sd21':
        return CondtionalUNet21()
    if type == 'sdxl':
        return CondtionalUNetXL()


def deepspeed_profile(model, args):
    flops, macs, params = get_model_profile(
        model=model,
        args=args,
        kwargs=None,
        print_profile=False,
        detailed=True,
        module_depth=-1,
        top_modules=1,
        warm_up=1,
        as_string=False,
        output_file=None,
        ignore_modules=None
    )
    return macs


def profile(type: str, prof_func):
    device = 'cpu'
    model = make_model(type)
    h, w = 64, 64
    bsz = 1

    latents = torch.randn(bsz, 4, h, w, device=device)
    timestep = torch.randint(0, 1000, (bsz, ), device=device)
    if type != 'sdxl':
        if type == 'sd15':
            condition = torch.randn(bsz, 77, 768, device=device)
        elif type == 'sd21':
            condition = torch.randn(bsz, 77, 1024, device=device)
        out = prof_func(model, args=(latents, timestep, condition))
        return f"{out/1e9:.3f} GMACs"
    condition = torch.randn(bsz, 77, 2048, device=device)
    time_ids = torch.randn(bsz, 6, device=device)
    text_embeds = torch.randn(bsz, 1280, device=device)
    out = prof_func(
        model, args=(latents, timestep,
                     condition, time_ids, text_embeds))
    return f"{out/1e9:.3f} GMACs"


if __name__ == '__main__':
    # print(profile('sd15', profile_macs))
    print(profile('sd21', profile_macs))
    # print(profile('sdxl', profile_macs))
    # print(profile('sd15', deepspeed_profile))
    print(profile('sd21', deepspeed_profile))
    # print(profile('sdxl', deepspeed_profile))
