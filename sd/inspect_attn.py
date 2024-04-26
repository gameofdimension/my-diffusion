import torch
from torch.utils import benchmark


def profile_model(fn, min_run_time=5):
    # https://github.com/facebookresearch/xformers/issues/678
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt='fn()',
        globals={"fn": fn},
        label="profile",
        sub_label="",
        description=""
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2 ** 20
    memory = f"Memory used: {memory} MB"
    print(res)
    print(memory)


def inspect(q, k, v):
    torch.nn.functional.scaled_dot_product_attention(q, k, v)


def main():
    for shape in [(160, 320, 64, 64), (160, 640, 32, 32), (160, 1280, 16, 16)]:
        n, c, h, w = shape
        q = torch.randn(shape, device='cuda').reshape(n, c, -1).transpose(1, 2)
        k = torch.randn(shape, device='cuda').reshape(n, c, -1).transpose(1, 2)
        v = torch.randn(shape, device='cuda').reshape(n, c, -1).transpose(1, 2)
        profile_model(lambda: inspect(
            q.reshape(n, h * w, 8, -1),
            k.reshape(n, h * w, 8, -1),
            v.reshape(n, h * w, 8, -1)
        ), min_run_time=5)
        profile_model(lambda: inspect(
            q.reshape(n, h * w, -1, 64),
            k.reshape(n, h * w, -1, 64),
            v.reshape(n, h * w, -1, 64)
        ), min_run_time=5)


if __name__ == "__main__":
    main()
