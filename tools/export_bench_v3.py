import argparse
import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"MBN3"
VERSION = 3

LAYER_TERNARY_3X3 = 0
LAYER_FP_3X3 = 1
LAYER_PIXELSHUFFLE_2X_SKIP = 2


class TernaryConv3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def quantize_params(self):
        scale = self.weight.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        w_scaled = self.weight / scale
        q = torch.round(w_scaled).clamp(-1, 1)
        w_dequant = q * scale
        return w_dequant, q.to(torch.int8), scale.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = self.quantize_params()
        return F.conv2d(x, w_q, self.bias, stride=1, padding=1)


class MyelinBenchNet(nn.Module):
    def __init__(self, hidden_ch: int = 32, hidden_layers: int = 3):
        super().__init__()
        layers = [TernaryConv3x3(3, hidden_ch)]
        for _ in range(hidden_layers - 1):
            layers.append(TernaryConv3x3(hidden_ch, hidden_ch))
        self.backbone = nn.ModuleList(layers)
        self.head = nn.Conv2d(hidden_ch, 12, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h = x
        for conv in self.backbone:
            h = F.relu(conv(h))
        return self.shuffle(self.head(h)) + x_up


def pack_ternary_per_oc(q_int8: torch.Tensor) -> np.ndarray:
    q_np = q_int8.cpu().numpy()
    out_ch, in_ch, kh, kw = q_np.shape
    assert kh == 3 and kw == 3
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


def make_test_image(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = xx / max(w - 1, 1)
    g = yy / max(h - 1, 1)
    b = (((xx // 16) + (yy // 16)) % 2).astype(np.float32) * 0.8 + 0.1
    return np.stack([r, g, b], axis=0).astype(np.float32)


def export_bench(out_dir: Path, in_w: int, in_h: int, hidden_ch: int, hidden_layers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    model = MyelinBenchNet(hidden_ch=hidden_ch, hidden_layers=hidden_layers).eval()
    with torch.no_grad():
        model.head.weight.mul_(0.05)
        model.head.bias.zero_()

    lr_fp32 = make_test_image(in_w, in_h)
    with torch.no_grad():
        hr_fp32 = model(torch.from_numpy(lr_fp32).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)

    lr_fp32.tofile(out_dir / "input_lr_fp32.bin")
    lr_fp32.astype(np.float16).tofile(out_dir / "input_lr_fp16.bin")
    hr_fp32.tofile(out_dir / "expected_hr_fp32.bin")

    layers = []
    ternary_words = []
    dense_fp32 = []
    scales = []
    biases = []

    for conv in model.backbone:
        w_dequant, q, s = conv.quantize_params()
        t_off = sum(len(a) for a in ternary_words)
        d_off = sum(len(a) for a in dense_fp32)
        s_off = sum(len(a) for a in scales)
        b_off = sum(len(a) for a in biases)

        ternary_words.append(pack_ternary_per_oc(q))
        dense_fp32.append(w_dequant.detach().cpu().numpy().astype(np.float32).reshape(-1))
        scales.append(s.detach().cpu().numpy().astype(np.float32))
        biases.append(conv.bias.detach().cpu().numpy().astype(np.float32))

        layers.append((
            LAYER_TERNARY_3X3, conv.weight.shape[1], conv.weight.shape[0], 1,
            t_off, d_off, s_off, b_off
        ))

    # 12-channel FP head
    d_off = sum(len(a) for a in dense_fp32)
    b_off = sum(len(a) for a in biases)
    dense_fp32.append(model.head.weight.detach().cpu().numpy().astype(np.float32).reshape(-1))
    biases.append(model.head.bias.detach().cpu().numpy().astype(np.float32))
    layers.append((LAYER_FP_3X3, model.head.in_channels, 12, 0, 0, d_off, 0, b_off))

    # 2x PixelShuffle + Global Bilinear Skip
    layers.append((LAYER_PIXELSHUFFLE_2X_SKIP, 12, 3, 0, 0, 0, 0, 0))

    t_arr = np.concatenate(ternary_words)
    d32_arr = np.concatenate(dense_fp32)
    d16_arr = d32_arr.astype(np.float16)
    s_arr = np.concatenate(scales)
    b_arr = np.concatenate(biases)

    # Pad FP16 blob to 4-byte boundary for clean D3D12 buffer sizing
    if len(d16_arr) % 2 != 0:
        d16_arr = np.concatenate([d16_arr, np.zeros(1, dtype=np.float16)])

    t_blob = t_arr.tobytes()
    d32_blob = d32_arr.tobytes()
    d16_blob = d16_arr.tobytes()
    s_blob = s_arr.tobytes()
    b_blob = b_arr.tobytes()

    bin_path = out_dir / "myelin_bench_v3.bin"
    with open(bin_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4I", VERSION, len(layers), in_w, in_h))
        for desc in layers:
            f.write(struct.pack("<8I", *desc))
        f.write(struct.pack("<5Q", len(t_blob), len(d32_blob), len(d16_blob), len(s_blob), len(b_blob)))
        f.write(t_blob)
        f.write(d32_blob)
        f.write(d16_blob)
        f.write(s_blob)
        f.write(b_blob)

    sparsity = np.mean([ (c.quantize_params()[1] == 0).float().mean().item() for c in model.backbone ]) * 100.0
    print(f"Exported {bin_path} | {in_w}x{in_h} -> {in_w*2}x{in_h*2} | {hidden_layers}x{hidden_ch}ch | Sparsity: {sparsity:.1f}%")
    print(f"Weights: Ternary={len(t_blob)/1024:.1f} KB | Dense FP16={len(d16_blob)/1024:.1f} KB | Dense FP32={len(d32_blob)/1024:.1f} KB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--out", type=str, default="outputs/bench_540p")
    args = p.parse_args()
    export_bench(Path(args.out), args.width, args.height, args.channels, args.layers)

