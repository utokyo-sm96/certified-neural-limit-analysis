"""Patch nlo_train.py with idea N2: fp16 autocast for the U-Net only (training); fields and losses in fp32."""
p = '/home/oldlinux/Research_Projects/Corrosion/PaperX/cnla/nlo_train.py'
s = open(p).read()
old = '''        out_v, out_s = model(features(mesh, th, L, W))
        ub, lb = bounds('''
assert old in s, "forward call not found"
s = s.replace(old, '''        with torch.autocast("cuda", dtype=torch.float16):               # N2: fp16 trunk
            out_v, out_s = model(features(mesh, th, L, W))
        out_v, out_s = out_v.float(), out_s.float()
        ub, lb = bounds(''')
old2 = '''        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()'''
assert old2 in s, "optimiser block not found"
s = s.replace(old2, '''        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()''')
old3 = '''    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)'''
assert old3 in s
s = s.replace(old3, old3 + '''
    scaler = torch.amp.GradScaler("cuda")''', 1)
open(p, 'w').write(s)
print("patched")
