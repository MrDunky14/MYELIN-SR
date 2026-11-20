import argparse
import math
import random
import struct
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MAGIC = b"MBN3"
VERSION = 3
LAYER_TERNARY_3X3 = 0
LAYER_FP_3X3 = 1
LAYER_PIXELSHUFFLE_2X_SKIP = 2


class Conv3x3Block(nn.Module):
    """Exact 3x3 conv matching myelin_v3_ocblock.hlsl in either Dense or 1.58-bit Ternary mode."""
    def __init__(self, in_ch: int, out_ch: int, ternary: bool):
        super().__init__()
        self.ternary = ternary
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def quantize_params(self):
        # Per-output-channel scale matching HLSL 1:1
        scale = self.weight.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        w_scaled = self.weight / scale
        q = torch.round(w_scaled).clamp(-1, 1)
        w_ste = (q - w_scaled).detach() + w_scaled
        return w_ste * scale, q.to(torch.int8), scale.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ternary:
            w, _, _ = self.quantize_params()
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, stride=1, padding=1)


class MyelinV3TrainNet(nn.Module):
    def __init__(self, hidden_ch: int = 32, hidden_layers: int = 3, ternary: bool = False):
        super().__init__()
        self.ternary = ternary
        layers = [Conv3x3Block(3, hidden_ch, ternary=ternary)]
        for _ in range(hidden_layers - 1):
            layers.append(Conv3x3Block(hidden_ch, hidden_ch, ternary=ternary))
        self.backbone = nn.ModuleList(layers)
        self.head = nn.Conv2d(hidden_ch, 12, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        # Small head init so model starts cleanly at the bilinear baseline
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h = x
        for conv in self.backbone:
            h = F.relu(conv(h))
        return self.shuffle(self.head(h)) + x_up


def make_lr_hr_pair(img_rgb: np.ndarray, scale: int = 2):
    """Modcrop HR to multiple of scale, generate LR via standard PIL Bicubic."""
    h, w, _ = img_rgb.shape
    h -= h % scale
    w -= w % scale
    hr = img_rgb[:h, :w, :]
    lr = np.array(Image.fromarray(hr).resize((w // scale, h // scale), Image.BICUBIC))
    return lr, hr


class SRPatchDataset(Dataset):
    def __init__(self, hr_dir: Path, patch_hr: int = 96, repeats: int = 20):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        paths = sorted([p for p in hr_dir.rglob("*") if p.suffix.lower() in exts])
        if not paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.images = []
        for p in paths:
            arr = np.array(Image.open(p).convert("RGB"))
            if arr.shape[0] >= patch_hr and arr.shape[1] >= patch_hr:
                self.images.append(arr)
        self.patch_hr = patch_hr
        self.repeats = repeats

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        h, w, _ = img.shape
        y = random.randint(0, h - self.patch_hr)
        x = random.randint(0, w - self.patch_hr)
        crop = img[y : y + self.patch_hr, x : x + self.patch_hr, :]

        # Standard SR dihedral augmentation (flip + 90-deg rotations)
        if random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1, :])
        if random.random() < 0.5:
            crop = np.ascontiguousarray(crop[::-1, :, :])
        k = random.randint(0, 3)
        if k:
            crop = np.ascontiguousarray(np.rot90(crop, k))

        lr, hr = make_lr_hr_pair(crop, scale=2)
        lr_t = torch.from_numpy(lr.transpose(2, 0, 1)).float() / 255.0
        hr_t = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
        return lr_t, hr_t


def rgb_to_y(img_01: np.ndarray) -> np.ndarray:
    """BT.601 Y-channel in [0, 255] matching standard SR benchmarks (BasicSR/MATLAB)."""
    return (16.0 + 65.481 * img_01[..., 0] + 128.553 * img_01[..., 1] + 24.966 * img_01[..., 2])


def calc_psnr(pred_01: np.ndarray, gt_01: np.ndarray, y_only: bool = True, shave: int = 2) -> float:
    pred = np.clip(pred_01, 0.0, 1.0)
    gt = np.clip(gt_01, 0.0, 1.0)
    if shave > 0:
        pred = pred[shave:-shave, shave:-shave, :]
        gt = gt[shave:-shave, shave:-shave, :]
    if y_only:
        diff = rgb_to_y(pred) - rgb_to_y(gt)
        mse = float(np.mean(diff ** 2))
        return 100.0 if mse < 1e-10 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    else:
        mse = float(np.mean((pred - gt) ** 2))
        return 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)


def calc_ssim_y(pred_01: np.ndarray, gt_01: np.ndarray, shave: int = 2) -> float:
    """Standard windowed Y-channel SSIM (11x11 Gaussian, sigma=1.5)."""
    pred_y = rgb_to_y(np.clip(pred_01, 0.0, 1.0))[shave:-shave, shave:-shave]
    gt_y = rgb_to_y(np.clip(gt_01, 0.0, 1.0))[shave:-shave, shave:-shave]
    p = torch.from_numpy(pred_y).float()[None, None]
    g = torch.from_numpy(gt_y).float()[None, None]

    coords = torch.arange(11, dtype=torch.float32) - 5.0
    k1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    k1d /= k1d.sum()
    win = (k1d[:, None] * k1d[None, :])[None, None]

    mu1 = F.conv2d(p, win)
    mu2 = F.conv2d(g, win)
    s1 = F.conv2d(p * p, win) - mu1 * mu1
    s2 = F.conv2d(g * g, win) - mu2 * mu2
    s12 = F.conv2d(p * g, win) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return float(ssim_map.mean().item())


@torch.no_grad()
def evaluate_folder(model: nn.Module, val_dir: Path, device: torch.device):
    model.eval()
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted([p for p in val_dir.rglob("*") if p.suffix.lower() in exts])
    metrics = {k: {"y_psnr": [], "rgb_psnr": [], "y_ssim": []} for k in ["bicubic", "lanczos", "model"]}

    for p in paths:
        hr_u8 = np.array(Image.open(p).convert("RGB"))
        lr_u8, hr_u8 = make_lr_hr_pair(hr_u8, scale=2)
        h, w, _ = hr_u8.shape
        gt_01 = hr_u8.astype(np.float32) / 255.0

        bic_01 = np.array(Image.fromarray(lr_u8).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0
        lan_01 = np.array(Image.fromarray(lr_u8).resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0

        lr_t = torch.from_numpy(lr_u8.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        sr_01 = model(lr_t).squeeze(0).clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)

        for name, pred in [("bicubic", bic_01), ("lanczos", lan_01), ("model", sr_01)]:
            metrics[name]["y_psnr"].append(calc_psnr(pred, gt_01, y_only=True, shave=2))
            metrics[name]["rgb_psnr"].append(calc_psnr(pred, gt_01, y_only=False, shave=2))
            metrics[name]["y_ssim"].append(calc_ssim_y(pred, gt_01, shave=2))

    return {
        k: {m: float(np.mean(v)) for m, v in d.items()}
        for k, d in metrics.items()
    }


def pack_ternary_per_oc(q_int8: torch.Tensor) -> np.ndarray:
    q_np = q_int8.cpu().numpy()
    out_ch, in_ch, _, _ = q_np.shape
    words_per_oc = (in_ch * 9 + 15) // 16
    packed = np.zeros((out_ch, words_per_oc), dtype=np.uint32)
    for oc in range(out_ch):
        flat = q_np[oc].reshape(-1)
        codes = np.zeros(words_per_oc * 16, dtype=np.uint32)
        codes[: len(flat)][flat == 1] = 1
        codes[: len(flat)][flat == -1] = 2
        for w in range(words_per_oc):
            word = np.uint32(0)
            base = w * 16
            for b in range(16):
                word |= codes[base + b] << np.uint32(b * 2)
            packed[oc, w] = word
    return packed.reshape(-1)


@torch.no_grad()
def export_to_dx12_bundle(model: MyelinV3TrainNet, sample_hr_path: Path, out_dir: Path):
    """Exports trained weights + a real test image into exact MBN3 format for dx12_bench_v3.exe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()

    hr_u8 = np.array(Image.open(sample_hr_path).convert("RGB"))
    lr_u8, hr_u8 = make_lr_hr_pair(hr_u8, scale=2)
    in_h, in_w, _ = lr_u8.shape

    lr_fp32 = lr_u8.transpose(2, 0, 1).astype(np.float32) / 255.0
    hr_pred = model(torch.from_numpy(lr_fp32).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)

    lr_fp32.tofile(out_dir / "input_lr_fp32.bin")
    lr_fp32.astype(np.float16).tofile(out_dir / "input_lr_fp16.bin")
    hr_pred.tofile(out_dir / "expected_hr_fp32.bin")
    Image.fromarray(lr_u8).save(out_dir / "sample_lr.png")
    Image.fromarray(hr_u8).save(out_dir / "sample_gt.png")
    Image.fromarray((np.clip(hr_pred.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        out_dir / "sample_pytorch_sr.png"
    )

    layers, ternary_words, dense_fp32, scales, biases = [], [], [], [], []
    for conv in model.backbone:
        t_off = sum(len(a) for a in ternary_words)
        d_off = sum(len(a) for a in dense_fp32)
        s_off = sum(len(a) for a in scales)
        b_off = sum(len(a) for a in biases)

        w_dequant, q, s = conv.quantize_params()
        ternary_words.append(pack_ternary_per_oc(q))
        # For Dense mode, export true continuous weights; for Ternary mode, export q*scale for exact parity
        w_dense = w_dequant if model.ternary else conv.weight
        dense_fp32.append(w_dense.detach().numpy().astype(np.float32).reshape(-1))
        scales.append(s.detach().numpy().astype(np.float32))
        biases.append(conv.bias.detach().numpy().astype(np.float32))
        layers.append((LAYER_TERNARY_3X3, conv.weight.shape[1], conv.weight.shape[0], 1, t_off, d_off, s_off, b_off))

    d_off = sum(len(a) for a in dense_fp32)
    b_off = sum(len(a) for a in biases)
    dense_fp32.append(model.head.weight.detach().numpy().astype(np.float32).reshape(-1))
    biases.append(model.head.bias.detach().numpy().astype(np.float32))
    layers.append((LAYER_FP_3X3, model.head.in_channels, 12, 0, 0, d_off, 0, b_off))
    layers.append((LAYER_PIXELSHUFFLE_2X_SKIP, 12, 3, 0, 0, 0, 0, 0))

    t_blob = np.concatenate(ternary_words).tobytes()
    d32_arr = np.concatenate(dense_fp32)
    d16_arr = d32_arr.astype(np.float16)
    if len(d16_arr) % 2 != 0:
        d16_arr = np.concatenate([d16_arr, np.zeros(1, dtype=np.float16)])

    with open(out_dir / "myelin_bench_v3.bin", "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4I", VERSION, len(layers), in_w, in_h))
        for desc in layers:
            f.write(struct.pack("<8I", *desc))
        f.write(struct.pack("<5Q", len(t_blob), d32_arr.nbytes, d16_arr.nbytes,
                            np.concatenate(scales).nbytes, np.concatenate(biases).nbytes))
        f.write(t_blob)
        f.write(d32_arr.tobytes())
        f.write(d16_arr.tobytes())
        f.write(np.concatenate(scales).tobytes())
        f.write(np.concatenate(biases).tobytes())
    print(f"[Export] Saved DX12 bundle to {out_dir} ({in_w}x{in_h} -> {in_w*2}x{in_h*2})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "ternary"], required=True)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Mode={args.mode} | Device={device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU 16 threads'})")

    ds = SRPatchDataset(args.train_dir, patch_hr=args.patch_size, repeats=args.repeats)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = MyelinV3TrainNet(args.channels, args.layers, ternary=(args.mode == "ternary")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_y_psnr = -1.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"myelin_v3_{args.mode}_best.pt"

    # Print exact dynamic baselines on val set first
    init_res = evaluate_folder(model, args.val_dir, device)
    print(f"[Baselines on {args.val_dir.name}] "
          f"Bicubic: {init_res['bicubic']['y_psnr']:.2f} dB (Y) / {init_res['bicubic']['y_ssim']:.4f} SSIM | "
          f"Lanczos: {init_res['lanczos']['y_psnr']:.2f} dB (Y) / {init_res['lanczos']['y_ssim']:.4f} SSIM")

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for lr_t, hr_t in loader:
            lr_t, hr_t = lr_t.to(device), hr_t.to(device)
            pred = model(lr_t)
            # Charbonnier L1 loss (standard for sharp SR convergence)
            loss = torch.sqrt((pred - hr_t) ** 2 + 1e-6).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
        sched.step()

        if ep % 10 == 0 or ep == args.epochs:
            res = evaluate_folder(model, args.val_dir, device)
            yp = res["model"]["y_psnr"]
            ys = res["model"]["y_ssim"]
            rp = res["model"]["rgb_psnr"]
            delta_bic = yp - res["bicubic"]["y_psnr"]
            delta_lan = yp - res["lanczos"]["y_psnr"]
            print(f"Ep {ep:3d}/{args.epochs} | Loss={loss_sum/len(loader):.5f} | "
                  f"Y-PSNR={yp:.2f} dB ({delta_bic:+.2f} vs Bic, {delta_lan:+.2f} vs Lan) | "
                  f"Y-SSIM={ys:.4f} | RGB-PSNR={rp:.2f} dB")
            if yp > best_y_psnr:
                best_y_psnr = yp
                torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    val_imgs = sorted([p for p in args.val_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg"}])
    sample_img = next((p for p in val_imgs if "butterfly" in p.name.lower()), val_imgs[0] if val_imgs else None)
    if sample_img:
        export_to_dx12_bundle(model, sample_img, args.out_dir / "dx12_bundle")


if __name__ == "__main__":
    main()
