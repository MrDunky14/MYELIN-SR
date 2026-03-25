"""
Loss functions for MYELIN-SR training.

Round 2 improvements — 3 losses working together:

  1. CHARBONNIER LOSS (replaces L1):
     sqrt((x-y)^2 + epsilon^2) — smoothed L1 that handles outlier pixels better.
     Does not saturate at large errors like MSE, not noisy at small errors like L1.

  2. FFT FREQUENCY LOSS (novel addition):
     Takes 2D FFT of SR and HR outputs, penalizes the *frequency spectrum difference*.
     L1-based pixel loss ignores high-frequency texture patterns completely.
     FFT loss directly supervises the network on sharpness and edge information —
     this is exactly where SR models fail (blurry output = missing high frequencies).

  3. VGG PERCEPTUAL LOSS (retained from Round 1):
     High-level semantic feature matching. Complements FFT (which is low-level).

  Why ternary networks benefit especially from FFT loss:
     With {-1, 0, +1} weights, each neuron responds to patterns via add/subtract.
     FFT loss creates gradient pressure toward high-frequency feature detectors,
     which naturally maps to ON-OFF center-surround patterns — the exact structure
     ternary {-1,+1} weights produce. It's a loss designed for our architecture.
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
        sr_features = self.vgg(self._normalize(sr.clamp(0, 1)))
        hr_features = self.vgg(self._normalize(hr.clamp(0, 1)))
        return F.l1_loss(sr_features, hr_features.detach())


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: smoothed L1 that behaves like L2 for small errors,
    L1 for large errors.

    L_charb(x, y) = sqrt((x-y)^2 + epsilon^2)

    Why better than L1 for SR:
    - L1 has a gradient discontinuity at 0 (causes training instability for ternary)
    - Charbonnier is smooth everywhere -> better gradient signal for quantized networks
    - Slightly more robust to outlier pixels than MSE
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon_sq = epsilon ** 2

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        diff_sq = (sr - hr) ** 2
        return torch.mean(torch.sqrt(diff_sq + self.epsilon_sq))


class FFTFrequencyLoss(nn.Module):
    """
    Frequency Spectrum Loss via 2D FFT.

    Key insight: Standard pixel losses (L1/L2) are equally weighted across all
    spatial frequencies. But SR is about recovering HIGH frequencies — fine
    texture, sharp edges, and detail.

    This loss computes the L1 difference of the 2D FFT magnitude spectrum,
    directly supervising the network to produce correct frequency content.

    Works in YCbCr space (Y channel only) to focus on luminance structure.

    Novel aspect: Exponential high-frequency emphasis — weight the spectrum
    inverse to frequency magnitude so rare high-freq components get more gradient.
    """

    def __init__(
        self,
        hf_emphasis: float = 2.0,   # exponent for high-frequency weighting
        use_log: bool = True,        # log-magnitude spectrum is more perceptually uniform
    ):
        super().__init__()
        self.hf_emphasis = hf_emphasis
        self.use_log = use_log

    def _rgb_to_y(self, img: torch.Tensor) -> torch.Tensor:
        """Extract luminance channel (Y) from RGB."""
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _make_hf_weight(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Create 2D spatial frequency weight map.
        Higher frequencies (closer to edges of FFT) get higher weight.
        """
        fy = torch.fft.fftfreq(h, device=device).abs()
        fx = torch.fft.fftfreq(w, device=device).abs()
        # Outer product: each position gets magnitude of its frequency
        freq_map = torch.sqrt(fy.unsqueeze(1) ** 2 + fx.unsqueeze(0) ** 2)
        # Normalize and apply emphasis
        freq_map = freq_map / freq_map.max().clamp_min(1e-8)
        weight = (1.0 + freq_map) ** self.hf_emphasis
        return weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_y = self._rgb_to_y(sr.clamp(0, 1))
        hr_y = self._rgb_to_y(hr.clamp(0, 1))

        H, W = sr_y.shape[-2], sr_y.shape[-1]

        # 2D FFT
        sr_fft = torch.fft.fft2(sr_y, norm="ortho")
        hr_fft = torch.fft.fft2(hr_y, norm="ortho")

        # Magnitude spectrum
        sr_mag = sr_fft.abs()
        hr_mag = hr_fft.abs()

        if self.use_log:
            sr_mag = torch.log1p(sr_mag)
            hr_mag = torch.log1p(hr_mag)

        # Weighted L1 on frequency spectrum
        weight = self._make_hf_weight(H, W, sr.device)
        loss = (weight * (sr_mag - hr_mag).abs()).mean()
        return loss


class MyelinSRLoss(nn.Module):
    """
    Combined training loss for MYELIN-SR Round 2.

    Total = Charbonnier + λ_freq × FFTFrequency + λ_perceptual × Perceptual

    Round 1 formula:  L1 + λ_perceptual × Perceptual
    Round 2 formula:  Charbonnier + λ_freq × FFT + λ_perceptual × Perceptual

    The FFT loss is the key new addition — it directly teaches the network to
    produce correct high-frequency content, which is exactly what SR needs.
    """

    def __init__(
        self,
        perceptual_weight: float = 0.1,
        freq_weight: float = 0.05,
        charbonnier_epsilon: float = 1e-3,
        use_perceptual: bool = True,
        use_freq: bool = True,
    ):
        super().__init__()
        self.pixel_loss      = CharbonnierLoss(epsilon=charbonnier_epsilon)
        self.perceptual_weight = perceptual_weight
        self.freq_weight     = freq_weight
        self.use_perceptual  = use_perceptual
        self.use_freq        = use_freq

        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()
        else:
            self.perceptual_loss = None

        if use_freq:
            self.freq_loss = FFTFrequencyLoss(hf_emphasis=2.0, use_log=True)
        else:
            self.freq_loss = None

    def forward(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: Combined scalar loss for backward pass
            loss_dict:  Individual loss components for logging
        """
        pixel = self.pixel_loss(sr, hr)
        loss_dict = {"charbonnier": pixel.item()}

        total = pixel

        if self.use_freq and self.freq_loss is not None:
            freq = self.freq_loss(sr, hr)
            total = total + self.freq_weight * freq
            loss_dict["fft_freq"] = freq.item()

        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(sr, hr)
            total = total + self.perceptual_weight * perceptual
            loss_dict["perceptual"] = perceptual.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
