"""
Phase 2: Temporal Training Script for MYELIN-SR

This script trains the MYELIN-Flow and TemporalFusion modules using the REDS 
dataset. It loads the pretrained Phase 1 spatial weights (myelin_sr_best.pt)
and freezes them, so only the temporal components learn to warp, mask, and blend.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from model import build_myelin_sr
from temporal_model import TemporalFusion
from temporal_loss import MyelinTemporalLoss
from temporal_dataset import build_temporal_loader

def train_temporal(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Temporal Training on: {device}")
    
    # 1. Load Pretrained Spatial Engine (Phase 1)
    print(f"Loading Phase 1 Spatial Backbone from {config['spatial_ckpt']}...")
    spatial_model_raw = build_myelin_sr(config['scale'], 'quality')
    
    ckpt = torch.load(config['spatial_ckpt'], map_location='cpu', weights_only=False)
    if 'model_state' in ckpt:
        # Load state dictated by the checkpoint format used in Kaggle train
        # Strip 'module.' prefix if it was saved via DataParallel
        state = {k.replace('module.', ''): v for k,v in ckpt['model_state'].items()}
        spatial_model_raw.load_state_dict(state)
    
    spatial_model_raw.eval()
    for param in spatial_model_raw.parameters():
        param.requires_grad = False  # Freeze spatial engine
        
    spatial_model = nn.DataParallel(spatial_model_raw).to(device)
    print("Spatial backbone loaded and frozen.")
    
    # 2. Initialize Temporal Fusion Engine (Phase 2)
    fusion_model = TemporalFusion(flow_search_radius=config['flow_radius']).to(device)
    fusion_model = nn.DataParallel(fusion_model) if torch.cuda.device_count() > 1 else fusion_model
    fusion_model.train()
    
    # 3. Optimizers & Losses
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=config['base_lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    criterion = MyelinTemporalLoss(
        spatial_weight=config['spatial_weight'],
        temp_consistency_weight=config['temporal_weight'],
        flow_warp_weight=config['flow_weight']
    ).to(device)
    
    # 4. Dataset
    print(f"Loading REDS Sequence Dataset from {config['data_dir']}...")
    train_loader = build_temporal_loader(
        root_dir=config['data_dir'],
        scale=config['scale'],
        patch_size=config['patch_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 5. Training Loop
    best_loss = float('inf')
    ckpt_dir = Path("/kaggle/working/temporal_checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(1, config['epochs'] + 1):
        ep_loss = 0.0
        ep_temp_loss = 0.0
        n_batches = 0
        
        for lr_seq, hr_seq in tqdm(train_loader, desc=f"Epoch {epoch:03d}"):
            # Sequences are stacked: [LR_prev, LR_cur]
            lr_prev = lr_seq[:, 0].to(device)
            lr_cur  = lr_seq[:, 1].to(device)
            
            hr_prev = hr_seq[:, 0].to(device)
            hr_cur  = hr_seq[:, 1].to(device)
            
            # Step A: Compute Spatial Output (Frozen)
            with torch.no_grad():
                sr_prev_base = spatial_model(lr_prev)
                sr_cur_base  = spatial_model(lr_cur)
                
            # Step B: Temporal Fusion & Flow Warp
            optimizer.zero_grad()
            fused_sr, flow_hr = fusion_model(sr_cur_base, sr_prev_base, lr_cur, lr_prev)
            
            # Step C: Compute Joint Loss
            loss, l_dict = criterion(fused_sr, sr_prev_base, hr_cur, hr_prev, flow_hr)
            
            loss.backward()
            nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()
            
            ep_loss += loss.item()
            ep_temp_loss += l_dict['temporal_consistency']
            n_batches += 1
            
        avg_loss = ep_loss / max(n_batches, 1)
        avg_temp = ep_temp_loss / max(n_batches, 1)
        
        print(f"Epoch {epoch:03d} | Total Loss: {avg_loss:.4f} | Temp Loss: {avg_temp:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            f_obj = fusion_model.module if isinstance(fusion_model, nn.DataParallel) else fusion_model
            torch.save({
                'epoch': epoch,
                'fusion_state': f_obj.state_dict(),
                'best_loss': best_loss,
            }, ckpt_dir / 'temporal_fusion_best.pt')
            print(f"  >> New Best Temporal Loss: {best_loss:.4f}")

if __name__ == "__main__":
    CONFIG = {
        'scale': 2,
        'epochs': 150,
        'batch_size': 8,
        'patch_size': 128,  # HR patch will be 256
        'num_workers': 4,
        'base_lr': 2e-4,
        'flow_radius': 4,
        'spatial_weight': 1.0,
        'temporal_weight': 0.5,
        'flow_weight': 1.0,
        'spatial_ckpt': 'outputs/myelin_sr_best.pt', # Generated from Phase 1
        'data_dir': '/kaggle/input/reds-dataset',
    }
    
    # Normally started from Kaggle Notebook, this allows direct debug
    # train_temporal(CONFIG)
