import sys, os
import cv2
import torch
import torchvision.transforms.functional as TF
from pathlib import Path

sys.path.insert(0, 'train')
from model import build_myelin_sr
from temporal_model import TemporalFusion

@torch.no_grad()
def process_video(input_path, output_path, spatial_ckpt, temporal_ckpt, upscale=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Models on {device}...")
    
    # 1. Spatial Engine
    spatial_engine = build_myelin_sr(upscale=upscale, quality_mode='quality').to(device)
    ckpt = torch.load(spatial_ckpt, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k,v in ckpt.get('model_state', ckpt).items()}
    spatial_engine.load_state_dict(state)
    spatial_engine.eval()
    
    # 2. Temporal Engine
    temporal_engine = TemporalFusion(flow_search_radius=4).to(device)
    ckpt2 = torch.load(temporal_ckpt, map_location='cpu', weights_only=False)
    state2 = {k.replace('module.', ''): v for k,v in ckpt2.get('fusion_state', ckpt2).items()}
    temporal_engine.load_state_dict(state2)
    temporal_engine.eval()
    
    # 3. Video Reader
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_width, out_height = width * upscale, height * upscale
    print(f"Processing Video: {width}x{height} -> {out_width}x{out_height} @ {fps}fps")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    prev_lr = None
    prev_sr = None
    
    print(f"Synthesizing {total_frames} frames...")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur_lr = TF.to_tensor(frame_rgb).unsqueeze(0).to(device)
        
        # Phase 1: Spatial SR
        cur_sr_base = spatial_engine(cur_lr)
        
        # Phase 2: Temporal Fusion
        if prev_lr is None:
            fused_sr = cur_sr_base
        else:
            fused_sr, _ = temporal_engine(
                sr_current=cur_sr_base,
                sr_history=prev_sr,
                lr_current=cur_lr,
                lr_history=prev_lr
            )
            
        prev_lr = cur_lr
        prev_sr = fused_sr
        
        # Convert back
        fused_sr_cpu = fused_sr.squeeze(0).clamp(0, 1).cpu()
        out_frame_rgb = TF.to_pil_image(fused_sr_cpu)
        import numpy as np
        out_frame_bgr = cv2.cvtColor(np.array(out_frame_rgb), cv2.COLOR_RGB2BGR)
        
        out.write(out_frame_bgr)
        count += 1
        if count % 10 == 0:
            print(f"Rendered {count}/{total_frames} frames...")
        
    cap.release()
    out.release()
    print(f"\nSaved stable 1440p video to {output_path}")

if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent.parent
    video_in = str(root_dir / "data" / "Cyberpunk_test_footage.mp4")
    if not os.path.exists(video_in):
        video_in = str(root_dir / "data" / "cyberpunk_temporal_test.mp4")
        
    video_out = str(root_dir / "data" / "MYELIN_STABLE_OUTPUT.mp4")
    spatial = str(root_dir / "outputs" / "myelin_sr_best.pt")
    temp = str(root_dir / "outputs" / "temporal_fusion_best.pt")
    
    process_video(video_in, video_out, spatial, temp)
