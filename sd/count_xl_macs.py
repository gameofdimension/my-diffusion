
def down_sample(h, w, cout):
    k = 3
    return (h/2)*(w/2)*cout*cout*(k*k)


def resblock(cin, cout, h, w):
    k = 3
    conv_shortcut = 0
    if cin != cout:
        conv_shortcut = h*w*cin*cout
    return h*w*cin*cout*k*k + h*w*cout*cout*k*k + conv_shortcut


def self_attn(cout, h, w, head):
    to_q = to_k = to_v = to_out = h*w*cout*cout
    attn = head*(h*w)**2*(cout/head)*2
    return to_q + to_k + to_v + to_out + attn


def cross_attn(cout, h, w, cl, cc, head):
    to_q = to_out = h*w*cout*cout
    to_k = to_v = cl*cc*cout
    attn = head*(h*w)*cl*(cout/head)*2
    return to_q + to_k + to_v + to_out + attn


def ff(cout, h, w):
    geglu = h*w*cout*8*cout
    out = h*w*cout*4*cout
    return geglu + out


def trans_block(cout, h, w, cl, cc, head):
    return self_attn(cout, h, w, head) + \
        cross_attn(cout, h, w, cl, cc, head) + ff(cout, h, w)


def trans_model(cout, h, w, cl, cc, head, trans_num: int):
    proj_in = h*w*cout*cout
    proj_out = h*w*cout*cout
    block = trans_block(cout, h, w, cl, cc, head)
    return (proj_in+proj_out+trans_num*block)


def crossdown(h, w, cin, cout, cl, cc, head, downsample: bool, trans_num: int):
    down = down_sample(h, w, cout)
    res1 = resblock(cin, cout, h, w)
    res2 = resblock(cout, cout, h, w)
    trans = trans_model(cout, h, w, cl, cc, head, trans_num)
    if downsample:
        down = down_sample(h, w, cout)
    else:
        down = 0
    return down+res1+res2+2*trans


def downblock(h, w, cin, cout, downsample: bool):
    res1 = resblock(cin, cout, h, w)
    res2 = resblock(cout, cout, h, w)
    if downsample:
        down = down_sample(h, w, cout)
    else:
        down = 0
    return res1+res2+down


def middleblock(h, w, cin, cout, cl, cc, head, trans_num):
    res1 = resblock(cin, cout, h, w)
    res2 = resblock(cout, cout, h, w)
    trans = trans_model(cout, h, w, cl, cc, head, trans_num=trans_num)
    return res1+trans+res2


def up_sample(h, w, cout):
    k = 3
    return (2*h)*(2*w)*cout*cout*(k*k)


def upblock(h, w, cin, cskip, cout, upsample: bool):
    res1 = resblock(cin+cskip, cout, h, w)
    res2 = resblock(cout+cskip, cout, h, w)
    res3 = resblock(cout+cskip, cout, h, w)
    if upsample:
        up = up_sample(h, w, cout)
    else:
        up = 0
    return res1+res2+res3+up


def cross_up(
        h, w, cin, cskip1, cskip2, cout, cl, cc, head,
        upsample: bool, trans_num: int):
    if upsample:
        up = up_sample(h, w, cout)
    else:
        up = 0
    res1 = resblock(cin+cskip1, cout, h, w)
    res2 = resblock(cout+cskip1, cout, h, w)
    res3 = resblock(cout+cskip2, cout, h, w)
    trans = trans_model(cout, h, w, cl, cc, head, trans_num=trans_num)

    return up+res1+res2+res3+3*trans


def the_downblock():
    h, w, cin, cout = 64, 64, 320, 320
    return downblock(h, w, cin, cout, downsample=True)


def crosdown1():
    h, w, cin, cout, cl = 32, 32, 320, 640, 77
    cc = 2048
    head = 10
    trans_num = 2
    return crossdown(
        h, w, cin, cout, cl, cc,
        head=head, downsample=True,
        trans_num=trans_num)


def crosdown2():
    h, w, cin, cout, cl = 16, 16, 640, 1280, 77
    cc = 2048
    head = 20
    trans_num = 10
    return crossdown(
        h, w, cin, cout, cl, cc,
        head=head, downsample=False,
        trans_num=trans_num)


def the_middleblock():
    h, w, cin, cout, cl = 16, 16, 1280, 1280, 77
    cc = 2048
    head = 20
    trans_num = 10
    return middleblock(h, w, cin, cout, cl, cc, head=head, trans_num=trans_num)


def crossup2():
    cc = 2048
    head = 20
    trans_num = 10
    h, w, cin, cskip1, cskip2, cout, cl = 16, 16, 1280, 1280, 640, 1280, 77  # noqa
    return cross_up(
        h, w, cin, cskip1, cskip2, cout, cl, cc,
        head=head, upsample=True,
        trans_num=trans_num)


def crossup1():
    cc = 2048
    head = 10
    trans_num = 2
    h, w, cin, cskip1, cskip2, cout, cl = 32, 32, 1280, 640, 320, 640, 77  # noqa
    return cross_up(
        h, w, cin, cskip1, cskip2, cout, cl, cc,
        head=head, upsample=True,
        trans_num=trans_num
    )


def th_upblock():
    h, w, cin, cskip, cout = 64, 64, 640, 320, 320
    return upblock(h, w, cin, cskip, cout, upsample=False)


def model():
    d1 = the_downblock()
    d2 = crosdown1()
    d3 = crosdown2()
    m = the_middleblock()
    u3 = crossup2()
    u2 = crossup1()
    u1 = th_upblock()

    return d1+d2+d3+m+u3+u2+u1


def main():
    macs = model()
    print(f"macs: {macs/1e9:.3f}")


if __name__ == "__main__":
    main()
