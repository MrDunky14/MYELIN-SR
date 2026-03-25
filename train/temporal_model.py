"""
MYELIN-Temporal: Deep Temporal Accumulation for MYELIN-SR

This module fuses the current SR output with the accumulated history from
previous frames. It uses MYELIN-Flow to align the history to the current frame.

Key Innovations:
1. ASTROCYTIC TEMPORAL GATING: Instead of a hard-coded alpha blend factor
   like TAA, we use a single FPSANConv2d layer to learn the blend weight.
   This leverages Astrocytic Stiffness — static regions freeze the weights
   to output high history dependence, while dynamic regions stay plastic
   and favor the current frame.
2. DISOCCLUSION DETECTION: History is discarded when the flow warp is invalid
   (e.g., when a foreground object moves, revealing a previously hidden background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fpsan_conv2d import FPSANConv2d
from flow_model import MyelinFlow

class TemporalFusion(nn.Module):
    """
    Fuses the high-resolution output of the current frame with the history.
    """
    def __init__(self, in_channels: int = 3, flow_search_radius: int = 4):
        super().__init__()
        
        # The lightweight optical flow backbone
        self.flow_estimator = MyelinFlow(search_radius=flow_search_radius)
        
        # Disocclusion Head: Detects when to completely discard history
        self.disocclusion_conv = FPSANConv2d(in_channels * 2 + 2, 16, kernel_size=3, padding=1)
        self.disocclusion_out  = FPSANConv2d(16, 1, kernel_size=3, padding=1)
        
        # Astrocytic Temporal Gating: Learns the blending alpha
        self.gating = FPSANConv2d(in_channels * 2 + 2, 1, kernel_size=3, padding=1)
        
    def _warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warps tensor x using optical flow.
        Flow is in pixel coordinates.
        """
        B, C, H, W = x.shape
        mesh_x, mesh_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        mesh_x = mesh_x.to(x.device).float().expand(B, -1, -1)
        mesh_y = mesh_y.to(x.device).float().expand(B, -1, -1)
        
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        new_x = mesh_x + u
        new_y = mesh_y + v
        
        new_x = 2.0 * new_x / max(W - 1, 1) - 1.0
        new_y = 2.0 * new_y / max(H - 1, 1) - 1.0
        grid = torch.stack((new_x, new_y), dim=-1)
        
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def forward(
        self, 
        sr_current: torch.Tensor, 
        sr_history: torch.Tensor, 
        lr_current: torch.Tensor, 
        lr_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sr_current: SR output for Frame t
        sr_history: Accumulated SR output from Frame t-1
        lr_current: Low-res Frame t (to estimate flow)
        lr_history: Low-res Frame t-1 (to estimate flow)
        
        Returns:
            fused_sr: Chronologically stable SR frame
            flow: Computed flow (for loss supervision)
        """
        B, C, H, W = sr_current.shape
        
        # 1. Estimate Flow (computed on low-res input for speed)
        # Flow vectors match the LR scale. We must upsample and scale them for HR.
        flow_lr = self.flow_estimator(lr_current, lr_history)
        scale_factor = float(sr_current.shape[-1]) / lr_current.shape[-1]
        
        flow_hr = F.interpolate(flow_lr, size=(H, W), mode='bilinear', align_corners=False)
        flow_hr = flow_hr * scale_factor
        
        # 2. Warp History SR to Current Frame Alignment
        warped_history = self._warp(sr_history, flow_hr)
        
        # 3. Disocclusion & Blending Features
        # Concat: Current SR + Warped History SR + HR Flow
        features = torch.cat([sr_current, warped_history, flow_hr], dim=1)
        
        # 4. Predict Disocclusion Mask [0, 1]
        # 1.0 = occluded (discard history), 0.0 = valid history
        d_mask = F.leaky_relu(self.disocclusion_conv(features), 0.1)
        d_mask = torch.sigmoid(self.disocclusion_out(d_mask))
        
        # 5. Astrocytic Temporal Gating (Alpha Blend)
        # 1.0 = use only current, 0.0 = use only history
        alpha = torch.sigmoid(self.gating(features))
        
        # 6. Apply Disocclusion: force alpha=1.0 where history is invalid
        alpha = torch.max(alpha, d_mask)
        
        # 7. Final Temporal Blend
        fused_sr = alpha * sr_current + (1.0 - alpha) * warped_history
        
        return fused_sr, flow_hr

    def print_architecture_summary(self):
        self.flow_estimator.print_architecture_summary()
        
        tp = sum(p.numel() for p in self.parameters())
        fp = sum(p.numel() for p in self.flow_estimator.parameters())
        fusion_params = tp - fp
        
        print(f"  Temporal Fusion Params: {fusion_params:,}")
        print(f"  Total Temporal Subsystem: {tp:,}")
        print("=" * 60)
