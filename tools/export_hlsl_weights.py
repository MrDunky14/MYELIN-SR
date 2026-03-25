"""
MYELIN-SR v2: Production-Grade HLSL Weight Exporter

Exports the full 37-layer PyTorch architecture into a GPU-ready binary with:
1. Global Header (magic, version, layer count)
2. Layer Descriptor Table (per-layer metadata for multi-pass dispatch)
3. Packed Ternary Weight Blob (16 weights per uint32)
4. FP32 Scale Blob (per output channel)
5. FP32 Bias Blob (per output channel)
6. FP32/FP16 Non-Ternary Weights (MoE router, Channel Attention, PixelShuffle)
"""

import os, sys, struct
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'train'))

from model import build_myelin_sr
from temporal_model import TemporalFusion
from fpsan_conv2d import FPSANConv2d

# ── Binary Format Constants ──
MAGIC = b'MYLN'
FORMAT_VERSION = 2

# Layer types
LAYER_FPSAN_CONV2D = 0   # Ternary convolution
LAYER_FP16_CONV2D  = 1   # FP16 reconstruction conv
LAYER_FP32_LINEAR  = 2   # Channel attention FC
LAYER_PIXELSHUFFLE = 3   # PixelShuffle (no weights, just metadata)
LAYER_MOE_ROUTER   = 4   # MoE routing conv (FP32)

def pack_ternary(weight_tensor):
    """Quantize to {-1,0,1} and pack 16 weights per uint32."""
    flat = weight_tensor.flatten().cpu()
    scale = flat.abs().mean().clamp_min(1e-5).item()
    ternary = torch.round(flat / scale).clamp(-1, 1).numpy().astype(np.int8)
    
    # Map: -1->2, 0->0, 1->1
    packed_vals = np.zeros_like(ternary, dtype=np.uint32)
    packed_vals[ternary == -1] = 2
    packed_vals[ternary == 1] = 1
    
    # Pad to multiple of 16
    n = len(packed_vals)
    pad_n = ((n + 15) // 16) * 16
    padded = np.zeros(pad_n, dtype=np.uint32)
    padded[:n] = packed_vals
    
    # Pack 16 per uint32
    result = []
    for i in range(0, pad_n, 16):
        word = 0
        for j in range(16):
            word |= int(padded[i + j]) << (j * 2)
        result.append(word)
    
    return np.array(result, dtype=np.uint32), scale

class BinaryWriter:
    def __init__(self, path):
        self.f = open(path, 'wb')
        self.layers = []
        self.weight_blobs = []
        self.scale_blobs = []
        self.bias_blobs = []
        self.fp_weight_blobs = []  # Non-ternary weights
        
    def add_fpsan_layer(self, name, module):
        """Add a ternary FPSANConv2d layer."""
        w = module.weight.data
        out_ch, in_ch, kH, kW = w.shape
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        pad = module.padding[0] if isinstance(module.padding, tuple) else module.padding
        groups = module.groups if hasattr(module, 'groups') else 1
        
        packed, scale = pack_ternary(w)
        
        # Per-output-channel scales
        scales = []
        for oc in range(out_ch):
            s = w[oc].abs().mean().clamp_min(1e-5).item()
            scales.append(s)
        scales = np.array(scales, dtype=np.float32)
        
        biases = module.bias.data.cpu().numpy().astype(np.float32) if module.bias is not None else np.zeros(out_ch, dtype=np.float32)
        
        w_offset = sum(b.nbytes for b in self.weight_blobs)
        s_offset = sum(b.nbytes for b in self.scale_blobs)
        b_offset = sum(b.nbytes for b in self.bias_blobs)
        
        self.layers.append({
            'name': name,
            'type': LAYER_FPSAN_CONV2D,
            'in_ch': in_ch * groups,  # actual input channels
            'out_ch': out_ch,
            'kernel': kH,
            'stride': stride,
            'padding': pad,
            'groups': groups,
            'w_offset': w_offset,
            'w_size': packed.nbytes,
            's_offset': s_offset,
            'b_offset': b_offset,
        })
        
        self.weight_blobs.append(packed)
        self.scale_blobs.append(scales)
        self.bias_blobs.append(biases)
        
        print(f"  [{len(self.layers)-1:2d}] TERNARY  {name:40s} | {in_ch*groups:3d}→{out_ch:3d} k={kH} s={stride} g={groups} | {packed.nbytes/1024:.2f} KB")
        
    def add_fp_layer(self, name, module, layer_type):
        """Add a non-ternary weight layer (FP16 or FP32)."""
        w = module.weight.data.cpu()
        out_ch = w.shape[0]
        in_ch = w.shape[1] if w.dim() > 1 else 1
        kH = w.shape[2] if w.dim() > 2 else 1
        stride = module.stride[0] if hasattr(module, 'stride') and isinstance(module.stride, tuple) else 1
        pad = module.padding[0] if hasattr(module, 'padding') and isinstance(module.padding, tuple) else 0
        
        # Store as FP16 for conv, FP32 for linear
        if layer_type == LAYER_FP16_CONV2D:
            w_bytes = w.half().numpy().tobytes()
        else:
            w_bytes = w.float().numpy().tobytes()
            
        biases = module.bias.data.cpu().numpy().astype(np.float32) if module.bias is not None else np.zeros(out_ch, dtype=np.float32)
        
        fp_offset = sum(len(b) for b in self.fp_weight_blobs)
        b_offset = sum(b.nbytes for b in self.bias_blobs)
        
        self.layers.append({
            'name': name,
            'type': layer_type,
            'in_ch': in_ch,
            'out_ch': out_ch,
            'kernel': kH,
            'stride': stride,
            'padding': pad,
            'groups': 1,
            'w_offset': fp_offset,
            'w_size': len(w_bytes),
            's_offset': 0,
            'b_offset': b_offset,
        })
        
        self.fp_weight_blobs.append(w_bytes)
        self.bias_blobs.append(biases)
        
        dtype_str = "FP16" if layer_type == LAYER_FP16_CONV2D else "FP32"
        print(f"  [{len(self.layers)-1:2d}] {dtype_str:8s} {name:40s} | {in_ch:3d}→{out_ch:3d} k={kH} | {len(w_bytes)/1024:.2f} KB")
        
    def finalize(self):
        """Write the complete binary file."""
        num_layers = len(self.layers)
        
        # ── Header ──
        self.f.write(MAGIC)
        self.f.write(struct.pack('<2I', FORMAT_VERSION, num_layers))
        
        # ── Layer Descriptor Table ──
        # Each descriptor: 11 x uint32 = 44 bytes
        for layer in self.layers:
            self.f.write(struct.pack('<11I',
                layer['type'],
                layer['in_ch'],
                layer['out_ch'],
                layer['kernel'],
                layer['stride'],
                layer['padding'],
                layer['groups'],
                layer['w_offset'],
                layer['w_size'],
                layer['s_offset'],
                layer['b_offset'],
            ))
        
        # ── Ternary Weight Blob ──
        ternary_blob_offset = self.f.tell()
        for blob in self.weight_blobs:
            self.f.write(blob.tobytes())
            
        # ── Scale Blob ──
        scale_blob_offset = self.f.tell()
        for blob in self.scale_blobs:
            self.f.write(blob.tobytes())
            
        # ── Bias Blob ──
        bias_blob_offset = self.f.tell()
        for blob in self.bias_blobs:
            self.f.write(blob.tobytes())
            
        # ── FP Weight Blob ──
        fp_blob_offset = self.f.tell()
        for blob in self.fp_weight_blobs:
            self.f.write(blob)
        
        # ── Footer: Blob Offsets ──
        self.f.write(struct.pack('<4Q',
            ternary_blob_offset,
            scale_blob_offset,
            bias_blob_offset,
            fp_blob_offset,
        ))
        
        total = self.f.tell()
        self.f.close()
        return total

def export_preset(quality_mode, spatial_ckpt=None, temporal_ckpt=None):
    device = torch.device('cpu')
    print(f"\n[{quality_mode.upper()} PRESET]")
    
    # Load Spatial Model
    spatial = build_myelin_sr(upscale=2, quality_mode=quality_mode).to(device)
    if spatial_ckpt and os.path.exists(spatial_ckpt):
        s_ckpt = torch.load(spatial_ckpt, map_location=device, weights_only=False)
        spatial.load_state_dict({k.replace('module.', ''): v for k,v in s_ckpt.get('model_state', s_ckpt).items()})
    else:
        print(f"  [Warning] No spatial weights found for {quality_mode}, exporting uninitialized!")
    spatial.eval()
    
    # Load Temporal Model
    temporal = TemporalFusion(flow_search_radius=4).to(device)
    if temporal_ckpt and os.path.exists(temporal_ckpt):
        t_ckpt = torch.load(temporal_ckpt, map_location=device, weights_only=False)
        # Note: temporal fusion arch doesn't change with quality_mode currently, but we load it anyway
        try:
            temporal.load_state_dict({k.replace('module.', ''): v for k,v in t_ckpt.get('fusion_state', t_ckpt).items()})
        except:
            print("  [Warning] Temporary temporal mismatch, exporting uninitialized!")
    else:
        print(f"  [Warning] No temporal weights found, exporting uninitialized!")
    temporal.eval()
    
    # Export
    out_path = str(ROOT / "outputs" / f"myelin_engine_v2_{quality_mode}.bin")
    writer = BinaryWriter(out_path)
    
    for name, module in spatial.named_modules():
        if isinstance(module, FPSANConv2d):
            writer.add_fpsan_layer(f"spatial.{name}", module)
        elif isinstance(module, torch.nn.Conv2d) and not isinstance(module, FPSANConv2d):
            writer.add_fp_layer(f"spatial.{name}", module, LAYER_FP16_CONV2D)
        elif isinstance(module, torch.nn.Linear):
            writer.add_fp_layer(f"spatial.{name}", module, LAYER_FP32_LINEAR)
            
    for name, module in temporal.named_modules():
        if isinstance(module, FPSANConv2d):
            writer.add_fpsan_layer(f"temporal.{name}", module)
            
    total_bytes = writer.finalize()
    print(f"  -> Exported {total_bytes/1024:.1f} KB to {out_path} ({len(writer.layers)} layers)")

def main():
    print("=" * 60)
    print("  MYELIN-SR v2: Production HLSL Weight Export")
    print("=" * 60)
    
    s_path = str(ROOT / "outputs" / "myelin_sr_best.pt")
    t_path = str(ROOT / "outputs" / "temporal_fusion_best.pt")
    
    # Export the primary high-quality tier
    export_preset('quality', s_path, t_path)
    
    # Export the maximum framerate tier
    export_preset('performance')

if __name__ == "__main__":
    main()
