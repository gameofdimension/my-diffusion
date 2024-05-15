
def t2i_config():
    return {
        # "name": 'uvit_t2i',
        "img_size": 32,
        "in_chans": 4,
        "patch_size": 2,
        "embed_dim": 512,
        "depth": 16,
        "num_heads": 8,
        "mlp_ratio": 4,
        "qkv_bias": False,
        "mlp_time_embed": False,
        "clip_dim": 768,
        "num_clip_token": 77,
    }
