import json

with open('outputs/training_log_r2.json', 'r') as f:
    log = json.load(f)

latest = log[-1]
best = max((e for e in log if 'val_psnr' in e), key=lambda x: x['val_psnr'])

print("--- FINAL EPOCH 300 ---")
print(f"Loss: {latest['loss']:.4f}")
print(f"PSNR: {latest['val_psnr']:.2f}")
print(f"SSIM: {latest['val_ssim']:.4f}")

print("\n--- BEST CHECKPOINT ---")
print(f"Epoch: {best['epoch']}")
print(f"PSNR: {best['val_psnr']:.2f} dB")
print(f"SSIM: {best['val_ssim']:.4f}")
print(f"LR: {best['lr']:.2e}")
