"""
Temporal Evaluation Suite for MYELIN-SR Phase 2

This script takes a folder containing consecutive low-resolution input frames
(e.g., from a gameplay capture) and processes them through the trained
Phase 1 (Spatial) and Phase 2 (TemporalFusion) engines.

It exports consecutive high-resolution frames, allowing for visual stability
tests (hunting for shimmering, ghosting, or jitter).
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, 'train')

from model import build_myelin_sr
from temporal_model import TemporalFusion

@torch.no_grad()
def evaluate_sequence(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Models on {device}...")
    
    # 1. Load Spatial Engine (Phase 1)
    spatial_engine = build_myelin_sr(scale=args.scale, arch_type='quality').to(device)
    if os.path.exists(args.spatial_ckpt):
        ckpt = torch.load(args.spatial_ckpt, map_location='cpu')
        state = {k.replace('module.', ''): v for k,v in ckpt['model_state'].items()}
        spatial_engine.load_state_dict(state)
        print(f"Loaded spatial backbone: {args.spatial_ckpt}")
    else:
        print(f"Warning: Spatial weights {args.spatial_ckpt} not found. Output will be noise.")
    
    spatial_engine.eval()
    
    # 2. Load Temporal Engine (Phase 2)
    temporal_engine = TemporalFusion(flow_search_radius=4).to(device)
    if os.path.exists(args.temporal_ckpt):
        ckpt = torch.load(args.temporal_ckpt, map_location='cpu')
        state = {k.replace('module.', ''): v for k,v in ckpt['fusion_state'].items()}
        temporal_engine.load_state_dict(state)
        print(f"Loaded temporal engine: {args.temporal_ckpt}")
    else:
        print(f"Warning: Temporal weights {args.temporal_ckpt} not found. Output will be unfused.")
    
    temporal_engine.eval()
    
    # 3. Setup Dataset Paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not frames:
        print(f"No valid image frames found in {input_dir}")
        return
        
    print(f"Found {len(frames)} frames. Processing temporal sequence...")
    
    # Track History state
    prev_lr = None
    prev_sr = None
    
    for i, frame_name in enumerate(tqdm(frames, desc="Generating Video Frames")):
        img_path = input_dir / frame_name
        img = Image.open(img_path).convert('RGB')
        
        # Add batch dim [1, C, H, W]
        cur_lr = TF.to_tensor(img).unsqueeze(0).to(device)
        
        # Spatial Step
        cur_sr_base = spatial_engine(cur_lr)
        
        # Temporal Step
        if i == 0:
            # First frame has no history, output standalone spatial frame
            fused_sr = cur_sr_base
        else:
            # Fuse with history
            fused_sr, _ = temporal_engine(
                sr_current=cur_sr_base,
                sr_history=prev_sr,
                lr_current=cur_lr,
                lr_history=prev_lr
            )
            
        # Update history buffer (Atomic Double Buffer concept)
        prev_lr = cur_lr
        prev_sr = fused_sr
        
        # Export frame
        fused_sr_cpu = fused_sr.squeeze(0).clamp(0, 1).cpu()
        out_img = TF.to_pil_image(fused_sr_cpu)
        
        out_name = f"fused_{i:04d}.png"
        out_img.save(output_dir / out_name)
        
    print(f"\\nPipeline Complete! Synthesized {len(frames)} temporally stable frames -> {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MYELIN-SR Phase 2 on sequential game frames.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of 720p sequential PNGs")
    parser.add_argument("--output_dir", type=str, default="data/benchmark_output", help="Where to save 1440p output frames")
    parser.add_argument("--spatial_ckpt", type=str, default="outputs/myelin_sr_best.pt", help="Phase 1 weights")
    parser.add_argument("--temporal_ckpt", type=str, default="outputs/temporal_fusion_best.pt", help="Phase 2 weights")
    parser.add_argument("--scale", type=int, default=2)
    
    # We allow running with defaults by faking args if empty (for IDE sanity)
    if len(sys.argv) == 1:
        print("Usage error: specify --input_dir and paths.")
    else:
        evaluate_sequence(parser.parse_args())
