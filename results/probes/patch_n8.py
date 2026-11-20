"""Patch nlo_train.py with idea N8: global self-attention (2 transformer layers) on the 16x5 U-Net bottleneck."""
p = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/nlo_train.py'
s = open(p).read()
old = '''        self.down = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(4)])'''
assert old in s
s = s.replace(old, old + '''
        nb = (NX // 16) * (NY // 16)                                     # bottleneck tokens (16 x 5 = 80)
        self.pos = nn.Parameter(torch.zeros(1, nb, ch[4]))
        layer = nn.TransformerEncoderLayer(d_model=ch[4], nhead=8, dim_feedforward=2 * ch[4], dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=2)          # N8: whole-plate context''')
old2 = '''        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)'''
assert old2 in s
s = s.replace(old2, '''        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2))
            skips.append(h)
        B, C, a, b = h.shape
        tok = h.flatten(2).transpose(1, 2)
        h = h + self.attn(tok + self.pos[:, :a * b]).transpose(1, 2).reshape(B, C, a, b) - tok.transpose(1, 2).reshape(B, C, a, b)
        skips[-1] = h''')
open(p, 'w').write(s)
print("patched")
