"""
MYELIN-SRNet — The Super Resolution Architecture

Spike-Routed Mixture of Experts with Combinatorial Texture Encoding,
Hybrid Precision (ternary backbone + FP16 reconstruction), and
Astrocytic Stiffness for continual learning.

Architecture Overview:
  Input (LR) → Shallow Feature Extraction (ternary)
    → Texture State Encoder (8 neurons, 255 combo states)
    → Spiking Router → Expert Selection
    → N Residual FP-SAN Blocks (ternary, per-expert)
    → FP16 PixelShuffle Reconstruction
    → Optional Ternary Refinement
    → Output (HR)

Architecture Decisions Reference:
  - Decision 1: Spike-Routed MoE (architecture_decisions.md)
  - Decision 2: Combinatorial Texture Encoding
  - Decision 3: Layered Spike Precision (ternary + FP16)
  - Decision 4: Dual Homeostatic Control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fpsan_conv2d import FPSANConv2d, FP16Conv2d


# ──────────────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Lightweight channel attention via global average pooling.
    
    Computes per-channel importance: GAP → FC(reduce) → ReLU → FC(expand) → Sigmoid
    Cost: O(C) — negligible compared to spatial convolutions.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualFPSANBlock(nn.Module):
    """
    Residual block using FPSANConv2d (ternary quantized).
    
    Structure:
      x → DepthwiseConv(ternary) → ReLU → PointwiseConv(ternary) → ChannelAttn → + x
    
    Depthwise separable design reduces FLOPs by ~8-9× vs standard conv.
    Ternary weights make it multiplication-free at inference.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = FPSANConv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = FPSANConv2d(
            channels, channels, kernel_size=1, padding=0
        )
        self.attention = ChannelAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.depthwise(x))
        out = self.pointwise(out)
        out = self.attention(out)
        return out + residual


# ──────────────────────────────────────────────────────────────────────
# Texture State Encoder (Decision 2: Combinatorial Texture Encoding)
# ──────────────────────────────────────────────────────────────────────

class TextureStateEncoder(nn.Module):
    """
    Encodes local texture information into a combinatorial spike chord.
    
    Uses 8 "texture neurons" — each fires (>threshold) or doesn't.
    8 neurons = 2^8 - 1 = 255 unique texture states.
    
    Inspired by Broca's vocal chord system (core_dump/FP-SAN-LLM/src/fpsan_v7_broca.cpp)
    but applied to visual texture classification.
    
    Output: soft chord (continuous 0-1 per neuron) for differentiable routing.
    """

    def __init__(self, in_channels: int, num_neurons: int = 8):
        super().__init__()
        self.num_neurons = num_neurons
        # Reduce spatial features to neuron activations
        self.encoder = nn.Sequential(
            FPSANConv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.ReLU(inplace=True),
            FPSANConv2d(in_channels, num_neurons, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: Texture state activations [B, num_neurons, H, W] in range (0, 1).
        Each spatial position gets its own texture chord.
        """
        return torch.sigmoid(self.encoder(x))


# ──────────────────────────────────────────────────────────────────────
# Spiking Router + Expert System (Decision 1: SR-MoE)
# ──────────────────────────────────────────────────────────────────────

class ExpertBlock(nn.Module):
    """Single expert: a small stack of Residual FP-SAN blocks."""

    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualFPSANBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SpikeRoutedMoE(nn.Module):
    """
    Spike-Routed Mixture of Experts.
    
    Instead of softmax routing (dense computation), texture state chords
    are used to compute soft expert weights via learned projections.
    
    During training: soft routing (all experts contribute, weighted).
    During inference: hard routing (top-1 expert only, for speed).
    
    Starting with 4 experts (Decision 1):
      Expert 0: Edge/boundary specialist
      Expert 1: Texture/pattern specialist  
      Expert 2: Flat/gradient specialist
      Expert 3: Fine detail specialist
    
    Experts specialize naturally through gradient flow — no manual assignment.
    """

    def __init__(self, channels: int, num_experts: int = 4, blocks_per_expert: int = 2):
        super().__init__()
        self.num_experts = num_experts

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertBlock(channels, blocks_per_expert) for _ in range(num_experts)
        ])

        # Router: texture state neurons → expert weights
        # Input: 8 texture neurons, output: num_experts weights
        self.router = nn.Sequential(
            nn.Conv2d(8, num_experts, kernel_size=1, bias=True),
        )

    def forward(
        self, x: torch.Tensor, texture_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Feature maps [B, C, H, W]
            texture_state: Texture chord [B, 8, H, W] from TextureStateEncoder
        Returns:
            Routed expert output [B, C, H, W]
        """
        # Compute routing weights from texture state
        # [B, num_experts, H, W]
        route_logits = self.router(texture_state)
        route_weights = torch.softmax(route_logits, dim=1)

        # Soft routing during training: weighted sum of all experts
        if self.training:
            output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x)
                weight = route_weights[:, i:i+1, :, :]  # [B, 1, H, W]
                output = output + expert_out * weight
            return output
        else:
            # Hard routing during inference: top-1 expert only
            top_expert = route_weights.argmax(dim=1)  # [B, H, W]

            output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                mask = (top_expert == i).unsqueeze(1).float()  # [B, 1, H, W]
                if mask.any():
                    expert_out = expert(x)
                    output = output + expert_out * mask
            return output


# ──────────────────────────────────────────────────────────────────────
# Main Architecture: MYELIN-SRNet
# ──────────────────────────────────────────────────────────────────────

class MyelinSRNet(nn.Module):
    """
    MYELIN-SR: Next-Gen Super Resolution powered by FP-SAN.
    
    Architecture:
      1. Shallow Feature Extraction (ternary FPSANConv2d)
      2. Texture State Encoding (combinatorial chord, 255 states)
      3. Spike-Routed MoE (4 specialized experts)
      4. FP16 PixelShuffle Reconstruction (color-precision critical)
      5. Optional Ternary Residual Refinement
    
    Unique capabilities vs DLSS:
      - Continual learning via Astrocytic Stiffness
      - Multiplication-free ternary inference
      - Per-game auto-adaptation
      - 0.1-0.5 MB deploy size
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        upscale_factor: int = 2,
        num_experts: int = 4,
        blocks_per_expert: int = 2,
        num_texture_neurons: int = 8,
        enable_refinement: bool = True,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.enable_refinement = enable_refinement

        # ── Stage 1: Shallow Feature Extraction (Ternary) ──
        self.shallow = FPSANConv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.shallow_relu = nn.ReLU(inplace=True)

        # ── Stage 2: Texture State Encoder ──
        self.texture_encoder = TextureStateEncoder(base_channels, num_texture_neurons)

        # ── Stage 3: Spike-Routed MoE ──
        self.moe = SpikeRoutedMoE(base_channels, num_experts, blocks_per_expert)

        # ── Stage 4: FP16 PixelShuffle Reconstruction (Decision 3) ──
        # This is the color-precision-critical layer — uses FP16, not ternary
        upscale_channels = in_channels * (upscale_factor ** 2)
        self.reconstruction = nn.Sequential(
            FP16Conv2d(base_channels, upscale_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

        # ── Stage 5: Optional Ternary Refinement (Quality Mode) ──
        if enable_refinement:
            self.refinement = nn.Sequential(
                FPSANConv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                FPSANConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            )
        else:
            self.refinement = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bilinear upscale for global residual learning
        x_up = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False
        )

        # Stage 1: Extract shallow features (ternary)
        feat = self.shallow_relu(self.shallow(x))

        # Stage 2: Encode texture state (combinatorial chord)
        texture_state = self.texture_encoder(feat)

        # Stage 3: Route through specialized experts
        feat = self.moe(feat, texture_state)

        # Stage 4: FP16 PixelShuffle reconstruction
        sr = self.reconstruction(feat)

        # Global residual: SR = upscaled_input + learned_residual
        sr = sr + x_up

        # Stage 5: Optional refinement pass
        if self.refinement is not None:
            sr = sr + self.refinement(sr)

        return sr

    def consolidate_all(self, rate: float = 0.08) -> None:
        """
        Sleep Consolidation across all FPSANConv2d layers.
        Call after a training phase to protect learned features.
        """
        for module in self.modules():
            if isinstance(module, FPSANConv2d):
                module.consolidate(rate)

    def get_total_sparsity(self) -> float:
        """Average sparsity across all ternary layers."""
        sparsities = []
        for module in self.modules():
            if isinstance(module, FPSANConv2d):
                sparsities.append(module.get_sparsity())
        return sum(sparsities) / max(len(sparsities), 1)

    def get_deploy_size_bytes(self) -> int:
        """Estimated deployment size with ternary packing."""
        ternary_bytes = 0
        fp_bytes = 0
        for name, module in self.named_modules():
            if isinstance(module, FPSANConv2d):
                # 2 bits per ternary weight, packed 16 per uint32
                n_weights = module.weight.numel()
                ternary_bytes += (n_weights * 2 + 31) // 32 * 4  # Pack into uint32s
                if module.bias is not None:
                    fp_bytes += module.bias.numel() * 4  # FP32 biases
                fp_bytes += 4  # Per-layer scale factor
            elif isinstance(module, nn.Conv2d):
                fp_bytes += module.weight.numel() * 2  # FP16 for reconstruction
                if module.bias is not None:
                    fp_bytes += module.bias.numel() * 4
            elif isinstance(module, nn.Linear):
                fp_bytes += module.weight.numel() * 2 + (module.bias.numel() * 4 if module.bias is not None else 0)

        return ternary_bytes + fp_bytes

    def print_architecture_summary(self) -> None:
        """Print a detailed summary of the architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        deploy_size = self.get_deploy_size_bytes()
        sparsity = self.get_total_sparsity()

        print("=" * 60)
        print("  MYELIN-SRNet Architecture Summary")
        print("=" * 60)
        print(f"  Upscale factor:    {self.upscale_factor}×")
        print(f"  Total parameters:  {total_params:,}")
        print(f"  Trainable params:  {trainable_params:,}")
        print(f"  Deploy size:       {deploy_size / 1024:.1f} KB ({deploy_size / (1024**2):.3f} MB)")
        print(f"  Ternary sparsity:  {sparsity * 100:.1f}%")
        print(f"  Refinement:        {'ON' if self.enable_refinement else 'OFF'}")
        print(f"  Experts:           {self.moe.num_experts}")
        print("=" * 60)


def build_myelin_sr(
    upscale: int = 2,
    quality_mode: str = "quality",
) -> MyelinSRNet:
    """
    Factory function with DLSS-style quality presets.
    
    Presets:
      - "performance": No refinement, 2 experts, minimal blocks. Fastest.
      - "balanced":    No refinement, 4 experts. Good trade-off.
      - "quality":     Full refinement, 4 experts. Best quality.
      - "ultra":       Full refinement, 4 experts, wider channels. Maximum quality.
    """
    configs = {
        "performance": dict(base_channels=32, num_experts=2, blocks_per_expert=1, enable_refinement=False),
        "balanced":    dict(base_channels=48, num_experts=4, blocks_per_expert=2, enable_refinement=False),
        "quality":     dict(base_channels=48, num_experts=4, blocks_per_expert=2, enable_refinement=True),
        "ultra":       dict(base_channels=64, num_experts=4, blocks_per_expert=3, enable_refinement=True),
    }

    if quality_mode not in configs:
        raise ValueError(f"Unknown quality mode: {quality_mode}. Choose from {list(configs.keys())}")

    return MyelinSRNet(upscale_factor=upscale, **configs[quality_mode])


if __name__ == "__main__":
    # Quick sanity check
    model = build_myelin_sr(upscale=2, quality_mode="quality")
    model.print_architecture_summary()

    # Test forward pass
    dummy_lr = torch.randn(1, 3, 64, 64)
    dummy_hr = model(dummy_lr)
    print(f"\n  Input:  {dummy_lr.shape}")
    print(f"  Output: {dummy_hr.shape}")
    assert dummy_hr.shape == (1, 3, 128, 128), f"Expected (1,3,128,128), got {dummy_hr.shape}"
    print("  ✅ Forward pass OK")
