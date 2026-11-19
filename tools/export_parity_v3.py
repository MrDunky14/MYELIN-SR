import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MAGIC = b"MYL3"
VERSION = 3

LAYER_TERNARY_3X3 = 0
LAYER_FP32_3X3 = 1
LAYER_PIXELSHUFFLE_2X_SKIP = 2


class TernaryConv3x3(nn.Module):
    """Per-output-channel scaled 1.58-bit ternary 3x3 conv (matches HLSL 1:1)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def quantize_params(self):
        # Exact per-output-channel scale: shape [OutC, 1, 1, 1]
        scale = self.weight.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
        w_scaled = self.weight / scale
        q = torch.round(w_scaled).clamp(-1, 1)
        w_ste = (q - w_scaled).detach() + w_scaled
        return w_ste * scale, q.to(torch.int8), scale.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, _, _ = self.quantize_params()
        return F.conv2d(x, w_q, self.bias, stride=1, padding=1)


class MyelinV3Net(nn.Module):
    def __init__(self, hidden_ch: int = 16):
        super().__init__()
        self.conv1 = TernaryConv3x3(3, hidden_ch)
        self.conv2 = TernaryConv3x3(hidden_ch, hidden_ch)
        # 12 output channels = 3 RGB * (2^2 subpixels)
        self.head = nn.Conv2d(hidden_ch, 12, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        res = self.shuffle(self.head(h))
        return res + x_up


def pack_ternary_per_oc(q_int8: torch.Tensor) -> np.ndarray:
    """
    Packs [OutC, InC, 3, 3] ternary weights into uint32 words.
    Each output channel is padded independently to a multiple of 16 weights:
      0 -> 0b00, +1 -> 0b01, -1 -> 0b10
    """
    q_np = q_int8.cpu().numpy()
    out_ch, in_ch, kh, kw = q_np.shape
    assert kh == 3 and kw == 3

    words_per_oc = (in_ch * 9 + 15) // 16
    packed = np.zeros((out_ch, words_per_oc), dtype=np.uint32)

    for oc in range(out_ch):
        flat = q_np[oc].reshape(-1)  # (ic, ky, kx) order
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


def make_test_image(w: int = 64, h: int = 64) -> np.ndarray:
    """Deterministic RGB CHW float32 [0, 1] image with gradients and sharp edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = xx / max(w - 1, 1)
    g = yy / max(h - 1, 1)
    b = (((xx // 8) + (yy // 8)) % 2).astype(np.float32) * 0.8 + 0.1
    return np.stack([r, g, b], axis=0).astype(np.float32)


def save_chw_png(chw: np.ndarray, path: Path):
    hwc = np.transpose(np.clip(chw, 0.0, 1.0), (1, 2, 0))
    img = (hwc * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def export_parity_bundle(out_dir: Path, in_w: int = 64, in_h: int = 64):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    model = MyelinV3Net(hidden_ch=16).eval()
    # Keep random head residual modest so test image stays visually readable
    with torch.no_grad():
        model.head.weight.mul_(0.05)
        model.head.bias.zero_()

    lr_np = make_test_image(in_w, in_h)
    with torch.no_grad():
        hr_np = model(torch.from_numpy(lr_np).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)

    # Save golden tensors + PNGs
    lr_np.tofile(out_dir / "input_lr.bin")
    hr_np.tofile(out_dir / "expected_hr.bin")
    save_chw_png(lr_np, out_dir / "input_lr.png")
    save_chw_png(hr_np, out_dir / "expected_pytorch.png")

    # Pack model layers
    layers = []
    ternary_words = []
    fp_weights = []
    scales = []
    biases = []

    def add_ternary(mod: TernaryConv3x3, activation: int):
        _, q, s = mod.quantize_params()
        w_off = sum(len(a) for a in ternary_words)
        s_off = sum(len(a) for a in scales)
        b_off = sum(len(a) for a in biases)
        packed = pack_ternary_per_oc(q)
        ternary_words.append(packed)
        scales.append(s.detach().cpu().numpy().astype(np.float32))
        biases.append(mod.bias.detach().cpu().numpy().astype(np.float32))
        layers.append((LAYER_TERNARY_3X3, mod.weight.shape[1], mod.weight.shape[0], activation, w_off, s_off, b_off, 0))

    def add_fp32(mod: nn.Conv2d, activation: int):
        w_off = sum(len(a) for a in fp_weights)
        b_off = sum(len(a) for a in biases)
        fp_weights.append(mod.weight.detach().cpu().numpy().astype(np.float32).reshape(-1))
        biases.append(mod.bias.detach().cpu().numpy().astype(np.float32))
        layers.append((LAYER_FP32_3X3, mod.in_channels, mod.out_channels, activation, w_off, 0, b_off, 0))

    add_ternary(model.conv1, activation=1)  # ReLU
    add_ternary(model.conv2, activation=1)  # ReLU
    add_fp32(model.head, activation=0)      # None
    layers.append((LAYER_PIXELSHUFFLE_2X_SKIP, 12, 3, 0, 0, 0, 0, 0))

    t_blob = np.concatenate(ternary_words).tobytes() if ternary_words else b""
    f_blob = np.concatenate(fp_weights).tobytes() if fp_weights else b""
    s_blob = np.concatenate(scales).tobytes() if scales else b""
    b_blob = np.concatenate(biases).tobytes() if biases else b""

    bin_path = out_dir / "myelin_v3.bin"
    with open(bin_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4I", VERSION, len(layers), in_w, in_h))
        for desc in layers:
            f.write(struct.pack("<8I", *desc))
        f.write(struct.pack("<4Q", len(t_blob), len(f_blob), len(s_blob), len(b_blob)))
        f.write(t_blob)
        f.write(f_blob)
        f.write(s_blob)
        f.write(b_blob)

    print(f"Exported {bin_path} ({bin_path.stat().st_size} bytes, {len(layers)} layers)")
    print(f"Golden HR range: [{hr_np.min():.4f}, {hr_np.max():.4f}]")


if __name__ == "__main__":
    export_parity_bundle(Path("outputs/parity_v3"))
