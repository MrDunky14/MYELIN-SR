import sys, os
sys.path.insert(0, os.path.join('.', 'train'))
import torch

results = []
for name, path in [
    ('epoch_50',  'outputs/myelin_sr_epoch_50.pt'),
    ('epoch_100', 'outputs/myelin_sr_epoch_100.pt'),
    ('epoch_150', 'outputs/myelin_sr_epoch_150.pt'),
    ('epoch_200', 'outputs/myelin_sr_epoch_200.pt'),
    ('best',      'outputs/myelin_sr_best.pt'),
]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    ep   = ckpt.get('epoch', '?')
    psnr = ckpt.get('best_psnr', 0.0)
    cfg  = ckpt.get('config', {})
    results.append(f"{name:>12}: epoch={str(ep):<4} best_psnr={psnr:.2f}dB  config={cfg}")

sz = os.path.getsize('outputs/myelin_sr_deploy.bin')
results.append(f"  deploy.bin: {sz/1024:.1f} KB")

output = "\n".join(results)
print(output)

with open('ckpt_summary.txt', 'w') as f:
    f.write(output + "\n")
