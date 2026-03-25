"""
Temporal Loss Functions for MYELIN-SR Phase 2

These losses supervise the MYELIN-Flow network and enforce temporal consistency
between consecutive super-resolved frames to prevent shimmering/flickering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CharbonnierLoss, FFTFrequencyLoss, VGGPerceptualLoss

def warp_image(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp an image using dense optical flow vectors.
    flow: [B, 2, H, W] in pixel displacement
    """
    B, C, H, W = x.shape
    mesh_x, mesh_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    mesh_x = mesh_x.to(x.device).float().expand(B, -1, -1)
    mesh_y = mesh_y.to(x.device).float().expand(B, -1, -1)
    
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    new_x = mesh_x + u
    new_y = mesh_y + v
    
    # Normalize to [-1, 1] for grid_sample
    new_x = 2.0 * new_x / max(W - 1, 1) - 1.0
    new_y = 2.0 * new_y / max(H - 1, 1) - 1.0
    grid = torch.stack((new_x, new_y), dim=-1)
    
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

class MyelinTemporalLoss(nn.Module):
    """
    Joint loss for Spatial SR + Temporal Consistency + Optical Flow.
    """
    def __init__(
        self, 
        spatial_weight: float = 1.0,
        temp_consistency_weight: float = 0.5,
        flow_warp_weight: float = 1.0,
    ):
        super().__init__()
        self.w_spatial = spatial_weight
        self.w_temp = temp_consistency_weight
        self.w_flow = flow_warp_weight
        
        self.charbonnier = CharbonnierLoss(epsilon=1e-3)
        self.fft = FFTFrequencyLoss()
        
    def forward(
        self, 
        sr_t: torch.Tensor, 
        sr_t_minus_1: torch.Tensor, 
        hr_t: torch.Tensor, 
        hr_t_minus_1: torch.Tensor,
        flow_hr: torch.Tensor
    ):
        """
        sr_t: Current frame SR output
        sr_t_minus_1: Previous frame SR output
        hr_t: Current ground truth HR
        hr_t_minus_1: Previous ground truth HR
        flow_hr: Estimated flow (upscaled to HR resolution)
        """
        # 1. Spatial SR Loss (Phase 1 standard)
        l_spatial_pixel = self.charbonnier(sr_t, hr_t)
        l_spatial_freq = self.fft(sr_t, hr_t)
        l_spatial = l_spatial_pixel + 0.05 * l_spatial_freq
        
        # 2. Flow Warp Loss (Supervises MYELIN-Flow unsupervisedly)
        # If flow is correct, warping HR_t-1 should look exactly like HR_t
        warped_hr = warp_image(hr_t_minus_1, flow_hr)
        l_flow = self.charbonnier(warped_hr, hr_t)
        
        # 3. Temporal Consistency Loss (Supervises the TemporalFusion blend)
        # Prevents flickering: the fused SR_t should naturally align with warped SR_t-1
        warped_sr = warp_image(sr_t_minus_1, flow_hr)
        l_temp = self.charbonnier(sr_t, warped_sr)
        
        # Total Joint Loss
        total_loss = (self.w_spatial * l_spatial) + \
                     (self.w_flow * l_flow) + \
                     (self.w_temp * l_temp)
                     
        loss_dict = {
            "spatial": l_spatial.item(),
            "flow_warp": l_flow.item(),
            "temporal_consistency": l_temp.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_dict
