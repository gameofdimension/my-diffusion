
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


def trans_model(cout, h, w, cl, cc, head):
    proj_in = h*w*cout*cout
    proj_out = h*w*cout*cout
    block = trans_block(cout, h, w, cl, cc, head)
    return (proj_in+proj_out+block)


def crossdown(h, w, cin, cout, cl, cc, head, downsample: bool):
    down = down_sample(h, w, cout)
    res1 = resblock(cin, cout, h, w)
    res2 = resblock(cout, cout, h, w)
    trans = trans_model(cout, h, w, cl, cc, head)
    if downsample:
        down = down_sample(h, w, cout)
    else:
        down = 0
    return down+res1+res2+trans+trans


def downblock(h, w, cin, cout, downsample: bool):
    res1 = resblock(cin, cout, h, w)
    res2 = resblock(cout, cout, h, w)
    if downsample:
        down = down_sample(h, w, cout)
    else:
        down = 0
    return res1+res2+down


def middleblock(h, w, cin, cout, cl, cc, head):
    res1 = resblock(cin, cout, h, w)
    res2 = resblock(cout, cout, h, w)
    trans = trans_model(cout, h, w, cl, cc, head)
    return res1+trans+res2


def up_sample(h, w, cout):
    k = 3
    return (2*h)*(2*w)*cout*cout*(k*k)


def upblock(h, w, cin, cskip, cout, upsample: bool):
    res = resblock(cin+cskip, cout, h, w)
    if upsample:
        up = up_sample(h, w, cout)
    else:
        up = 0
    return 3*res+up


def cross_up(h, w, cin, cskip1, cskip2, cout, cl, cc, head, upsample: bool):
    if upsample:
        up = up_sample(h, w, cout)
    else:
        up = 0
    res1 = resblock(cin+cskip1, cout, h, w)
    res2 = resblock(cout+cskip1, cout, h, w)
    res3 = resblock(cout+cskip2, cout, h, w)
    trans = trans_model(cout, h, w, cl, cc, head)
    return up+res1+res2+res3+3*trans


def crosdown1(kind):
    h, w, cin, cout, cl = 64, 64, 320, 320, 77
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    return crossdown(h, w, cin, cout, cl, cc, head=head, downsample=True)


def crosdown2(kind):
    h, w, cin, cout, cl = 32, 32, 320, 640, 77
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    return crossdown(h, w, cin, cout, cl, cc, head=head, downsample=True)


def crosdown3(kind):
    h, w, cin, cout, cl = 16, 16, 640, 1280, 77
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    return crossdown(h, w, cin, cout, cl, cc, head=head, downsample=True)


def the_downblock():
    h, w, cin, cout = 8, 8, 1280, 1280
    return downblock(h, w, cin, cout, downsample=False)


def the_middleblock(kind):
    h, w, cin, cout, cl = 8, 8, 1280, 1280, 77
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    return middleblock(h, w, cin, cout, cl, cc, head=head)


def th_upblock():
    h, w, cin, cskip, cout = 8, 8, 1280, 1280, 1280
    return upblock(h, w, cin, cskip, cout, upsample=True)


def crossup3(kind):
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    h, w, cin, cskip1, cskip2, cout, cl = 16, 16, 1280, 1280, 640, 1280, 77  # noqa
    return cross_up(h, w, cin, cskip1, cskip2, cout, cl, cc, head=head, upsample=True)  # noqa


def crossup2(kind):
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    h, w, cin, cskip1, cskip2, cout, cl = 32, 32, 1280, 640, 320, 640, 77  # noqa
    return cross_up(h, w, cin, cskip1, cskip2, cout, cl, cc, head=head, upsample=True)  # noqa


def crossup1(kind):
    if kind == '15':
        cc = 768
        head = 8
    elif kind == '21':
        cc = 1024
        head = 5
    h, w, cin, cskip1, cskip2, cout, cl = 64, 64, 640, 320, 320, 320, 77  # noqa
    return cross_up(h, w, cin, cskip1, cskip2, cout, cl, cc, head=head, upsample=False)  # noqa


def model(kind):
    d1 = crosdown1(kind)
    d2 = crosdown2(kind)
    d3 = crosdown3(kind)
    d4 = the_downblock()
    m = the_middleblock(kind)
    u4 = th_upblock()
    u3 = crossup3(kind)
    u2 = crossup2(kind)
    u1 = crossup1(kind)

    return d1+d2+d3+d4+m+u4+u3+u2+u1


def main():
    macs = model('15')
    print(f"macs: {macs/1e9:.3f}")
    macs = model('21')
    print(f"macs: {macs/1e9:.3f}")


if __name__ == "__main__":
    main()
