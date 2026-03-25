"""
Loss functions for MYELIN-SR training.

L1 (Pixel Loss) + VGG Perceptual Loss.
GAN loss optional for future phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Compares high-level features between SR output and HR ground truth.
    Uses relu3_3 features (layer index 16) — balances texture and structure.
    
    Frozen VGG (no gradients) — only used as a feature extractor.
    """

    def __init__(self, feature_layer: int = 16):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:feature_layer + 1]

        # Freeze all VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet statistics."""
        return (x - self.mean) / self.std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_features = self.vgg(self._normalize(sr))
        hr_features = self.vgg(self._normalize(hr))
        return F.l1_loss(sr_features, hr_features)


class MyelinSRLoss(nn.Module):
    """
    Combined training loss for MYELIN-SR.
    
    Total = L1_pixel + λ_perceptual × Perceptual
    
    L1 preserves brightness and prevents color shift.
    Perceptual adds texture quality and sharpness.
    """

    def __init__(self, perceptual_weight: float = 0.1, use_perceptual: bool = True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual

        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: Combined scalar loss for backward pass
            loss_dict: Individual loss components for logging
        """
        l1 = self.l1_loss(sr, hr)
        loss_dict = {"l1": l1.item()}

        total = l1

        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(sr, hr)
            total = total + self.perceptual_weight * perceptual
            loss_dict["perceptual"] = perceptual.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
