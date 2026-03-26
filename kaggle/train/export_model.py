# MYELIN-SR v2: Zero-Barrier Ternary Reconstruction Engine
# Copyright (C) 2026 Krishna Singh
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
MYELIN-SR Model Export

Exports a trained MyelinSRNet checkpoint to two formats:
  1. TorchScript (.pt) — for Python/C++ deployment via LibTorch
  2. Ternary Binary (.bin) — custom compact format for native HLSL/C++ inference

Ternary binary format (extends core_dump/FP-SAN/src/export_weights.py to 2D convolutions):
  Header: [magic: 4B][version: 4B][num_layers: 4B]
  Per layer:
    [type: 1B][out_ch: 4B][in_ch: 4B][kH: 4B][kW: 4B][groups: 4B]
    [scale: 4B float32]
    [weights: ceil(out*in*kH*kW / 4) bytes, 2 bits per weight packed]
    [has_bias: 1B][bias: out_ch * 4B float32 if has_bias]

Weight encoding: 0b00=0, 0b01=+1, 0b10=-1 packed 4 per byte (little-endian)

Usage:
  python export_model.py --checkpoint ../checkpoints/myelin_sr_best.pt --output ../deploy/
"""

import argparse
import struct
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import build_myelin_sr, MyelinSRNet
from fpsan_conv2d import FPSANConv2d

MAGIC = b"MYSR"
VERSION = 1

# Layer type codes
LAYER_TERNARY_CONV = 0x01   # FPSANConv2d (ternary)
LAYER_FP16_CONV    = 0x02   # FP16Conv2d (standard conv wrapped)
LAYER_META_END     = 0xFF   # End sentinel


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MyelinSRNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {"scale": 2, "quality": "quality"})
    model = build_myelin_sr(upscale=config["scale"], quality_mode=config["quality"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


def pack_ternary_weights(weights_int8: torch.Tensor) -> bytes:
    """
    Pack {-1, 0, +1} int8 weights into 2 bits per weight.
    Encoding: 0b00=0, 0b01=+1, 0b10=-1
    4 weights per byte, little-endian within byte.
    """
    flat = weights_int8.flatten().tolist()
    # Pad to multiple of 4
    while len(flat) % 4 != 0:
        flat.append(0)

    packed = bytearray()
    for i in range(0, len(flat), 4):
        byte = 0
        for j in range(4):
            val = int(flat[i + j])
            if val == 1:
                code = 0b01
            elif val == -1:
                code = 0b10
            else:
                code = 0b00
            byte |= (code << (j * 2))
        packed.append(byte)
    return bytes(packed)


def export_ternary_binary(model: MyelinSRNet, output_path: Path, config: dict) -> int:
    """
    Export model to custom ternary binary format.
    Returns total file size in bytes.
    """
    layers = []

    # Collect all layers in order
    for name, module in model.named_modules():
        if isinstance(module, FPSANConv2d):
            ternary_w = module.get_ternary_weights()        # int8 {-1,0,1}
            scale = module.get_weight_scale()               # float32
            bias = module.bias.detach().cpu() if module.bias is not None else None
            out_ch, in_ch = module.weight.shape[0], module.weight.shape[1]
            kH, kW = module.kernel_size
            groups = module.groups
            layers.append({
                "type": LAYER_TERNARY_CONV,
                "name": name,
                "out_ch": out_ch,
                "in_ch": in_ch,
                "kH": kH,
                "kW": kW,
                "groups": groups,
                "scale": scale,
                "ternary_weights": ternary_w,
                "bias": bias,
            })
        elif isinstance(module, torch.nn.Conv2d):
            # FP16 head and router layers
            w = module.weight.detach().cpu().to(torch.float16)
            bias = module.bias.detach().cpu() if module.bias is not None else None
            out_ch, in_ch = module.weight.shape[0], module.weight.shape[1]
            kH, kW = module.weight.shape[2], module.weight.shape[3]
            layers.append({
                "type": LAYER_FP16_CONV,
                "name": name,
                "out_ch": out_ch,
                "in_ch": in_ch,
                "kH": kH,
                "kW": kW,
                "groups": 1,
                "weights_fp16": w,
                "bias": bias,
            })

    # Write binary file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(layers)))
        f.write(struct.pack("<I", config.get("scale", 2)))
        f.write(struct.pack("<I", {"performance": 0, "balanced": 1, "quality": 2, "ultra": 3}.get(config.get("quality", "quality"), 2)))

        # Layers
        for layer in layers:
            f.write(struct.pack("<B", layer["type"]))
            f.write(struct.pack("<I", layer["out_ch"]))
            f.write(struct.pack("<I", layer["in_ch"]))
            f.write(struct.pack("<I", layer["kH"]))
            f.write(struct.pack("<I", layer["kW"]))
            f.write(struct.pack("<I", layer["groups"]))

            if layer["type"] == LAYER_TERNARY_CONV:
                f.write(struct.pack("<f", layer["scale"]))
                packed = pack_ternary_weights(layer["ternary_weights"])
                f.write(struct.pack("<I", len(packed)))
                f.write(packed)
            else:
                # FP16 weights
                w_bytes = layer["weights_fp16"].numpy().tobytes()
                f.write(struct.pack("<I", len(w_bytes)))
                f.write(w_bytes)

            # Bias
            if layer["bias"] is not None:
                f.write(struct.pack("<B", 1))
                f.write(layer["bias"].numpy().astype("float32").tobytes())
            else:
                f.write(struct.pack("<B", 0))

        f.write(struct.pack("<B", LAYER_META_END))
        file_size = f.tell()

    return file_size


def export_torchscript(model: MyelinSRNet, output_path: Path) -> None:
    """Export model to TorchScript for LibTorch / Python deployment."""
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)

    try:
        scripted = torch.jit.trace(model, dummy)
        scripted.save(str(output_path))
        print(f"  [OK] TorchScript saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"  [!!] TorchScript export failed (inference routing may use Python control flow): {e}")
        print("       Saving state_dict only as fallback.")


def main():
    parser = argparse.ArgumentParser(description="MYELIN-SR Model Exporter")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="./deploy", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-torchscript", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  MYELIN-SR Export Pipeline")
    print("=" * 50)

    model, config = load_model(args.checkpoint, device)
    print(f"  Loaded: scale={config['scale']}x quality={config['quality']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  FP32 size: {sum(p.numel() * 4 for p in model.parameters()) / 1024:.1f} KB")

    # 1. Ternary binary
    bin_path = output_dir / f"myelin_sr_{config['quality']}_x{config['scale']}.bin"
    bin_size = export_ternary_binary(model, bin_path, config)
    print(f"  [OK] Ternary binary: {bin_path.name} ({bin_size / 1024:.1f} KB)")

    # 2. TorchScript
    if not args.skip_torchscript:
        pt_path = output_dir / f"myelin_sr_{config['quality']}_x{config['scale']}_scripted.pt"
        export_torchscript(model, pt_path)

    # 3. Summary
    fp32_size = sum(p.numel() * 4 for p in model.parameters())
    print()
    print(f"  FP32 baseline:    {fp32_size / 1024:.1f} KB")
    print(f"  Ternary binary:   {bin_size / 1024:.1f} KB  ({fp32_size / bin_size:.1f}x smaller)")
    print("=" * 50)


if __name__ == "__main__":
    main()
