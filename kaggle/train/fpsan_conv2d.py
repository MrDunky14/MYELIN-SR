"""
FPSANConv2d — The Core Building Block of MYELIN-SR

Extends Krishna's FPSANLinear (core_dump/FP-SAN/src/fpsan_proofsheet.py) to 2D convolutions.

Key Mechanics:
  1. TERNARY FORWARD PASS: Weights are FP32 during training but quantized to {-1, 0, +1}
     inside the forward pass via round(w/scale).clamp(-1,1)*scale. Gradients flow through
     the full-precision master weights (Straight-Through Estimator).

  2. ASTROCYTIC STIFFNESS: Per-weight consolidation buffer that tracks activity.
     Heavily-used weights become "stiff" (protected from future updates), solving
     catastrophic forgetting. This is the FP-SAN equivalent of EWC but simpler.

  3. SLEEP CONSOLIDATION: Converts accumulated daily_activity into permanent stiffness,
     then resets activity. Hardware equivalent of biological memory consolidation.

  4. MULTIPLICATION-FREE INFERENCE: At export time, weights are quantized to int8 {-1,0,1}.
     C++/HLSL inference uses only add/subtract branching — zero FP multiplications.

References:
  - FPSANLinear: core_dump/FP-SAN/src/fpsan_proofsheet.py L105-L148
  - Export pipeline: core_dump/FP-SAN/src/export_weights.py
  - C++ inference: core_dump/FP-SAN/src/fpsan_infer.cpp L62-L96
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FPSANConv2d(nn.Module):
    """
    Ternary-Quantized 2D Convolution with Astrocytic Stiffness.
    
    Drop-in replacement for nn.Conv2d. Trains in FP32, forward-passes in ternary.
    Supports depthwise separable mode via groups parameter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = True,
        min_plasticity: float = 0.15,
        stiffness_cap: float = 6.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.min_plasticity = min_plasticity
        self.stiffness_cap = stiffness_cap

        # FP32 master weights (training state)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Astrocytic Stiffness: per-weight consolidation buffer
        self.register_buffer(
            "stiffness", torch.ones_like(self.weight.data)
        )
        # Daily activity tracker (accumulated between sleep phases)
        self.register_buffer(
            "daily_activity", torch.zeros_like(self.weight.data)
        )

        # Input trace for forward-mode learning (DFA)
        self.trace_input: torch.Tensor | None = None

        # Initialize weights (Kaiming uniform, as standard for conv layers)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels // groups * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize_ternary(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Ternary quantization: round to {-1, 0, +1} scaled by mean absolute value.
        
        This is Krishna's original technique from FPSANLinear:
            scale = weight.abs().mean().clamp_min(1e-5)
            w_quant = round(weight / scale).clamp(-1, 1) * scale
        
        During training, gradients flow through the FP32 master weights via
        the Straight-Through Estimator (STE) — the quantization is treated
        as identity in the backward pass.
        """
        scale = weight.detach().abs().mean().clamp_min(1e-5)
        # Forward: quantized. Backward: straight-through to FP32 weights.
        w_scaled = weight / scale
        w_ternary = (torch.round(w_scaled).clamp(-1, 1) - w_scaled).detach() + w_scaled
        return w_ternary * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input trace for potential DFA update
        if self.training:
            self.trace_input = x.detach()

        # Quantize weights to ternary in forward pass
        w_q = self._quantize_ternary(self.weight)

        return F.conv2d(
            x, w_q, self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

    @torch.no_grad()
    def apply_forward_update(
        self,
        local_error: torch.Tensor,
        lr: float,
        sample_gate: torch.Tensor | None = None,
    ) -> None:
        """
        Direct Feedback Alignment (DFA) update — backprop-free.
        
        Uses the stored input trace and a local error signal to update weights
        without computing a full backward pass through the network.
        
        For convolutional layers, this computes the correlation between input
        patches and error signals, gated by astrocytic stiffness.
        
        References: fpsan_proofsheet.py L124-L142
        """
        if self.trace_input is None:
            return

        # Compute weight gradient via correlation of input and error
        # For conv layers, we use F.conv2d in "backward" configuration
        batch_size = local_error.size(0)

        # Unfold input to get patches matching the kernel
        # This is equivalent to the gradient computation for conv2d
        x = self.trace_input
        if sample_gate is not None:
            gate = sample_gate.view(batch_size, 1, 1, 1)
            local_error = local_error * gate
            normalizer = gate.sum().clamp_min(1.0)
        else:
            normalizer = float(batch_size)

        # Compute raw weight update (gradient approximation)
        # Using the efficient method: grad_w = conv2d(input^T, error^T)
        raw_weight_grad = torch.zeros_like(self.weight)
        for b in range(batch_size):
            # Per-sample gradient accumulation (memory-efficient)
            x_b = x[b:b+1]
            e_b = local_error[b:b+1]
            # Standard convolution gradient formula
            for g in range(self.groups):
                c_in = self.in_channels // self.groups
                c_out = self.out_channels // self.groups
                x_g = x_b[:, g*c_in:(g+1)*c_in]
                e_g = e_b[:, g*c_out:(g+1)*c_out]
                for co in range(c_out):
                    for ci in range(c_in):
                        raw_weight_grad[g*c_out+co, ci] += F.conv2d(
                            x_g[:, ci:ci+1],
                            e_g[:, co:co+1].flip(-1, -2),
                            padding=self.kernel_size[0] - 1
                        )[0, 0, :self.kernel_size[0], :self.kernel_size[1]]

        raw_weight_grad /= normalizer

        # Apply Astrocytic Stiffness gating
        normalized_stiffness = self.stiffness / self.stiffness.max().clamp_min(1.0)
        plasticity_mask = (1.0 - normalized_stiffness).clamp(self.min_plasticity, 1.0)

        # Update FP32 master weights (not the quantized ones)
        self.weight.data -= lr * (raw_weight_grad * plasticity_mask / self.stiffness.clamp_min(1.0))

        if self.bias is not None:
            raw_bias_grad = local_error.mean(dim=(0, 2, 3))
            self.bias.data -= lr * raw_bias_grad

        # Track activity for sleep consolidation
        self.daily_activity += raw_weight_grad.abs()
        self.trace_input = None

    @torch.no_grad()
    def consolidate(self, rate: float = 0.08) -> None:
        """
        Sleep Consolidation: Convert daily activity into permanent stiffness.

        High-activity weights become stiffer (harder to modify), protecting
        learned features from catastrophic forgetting during new task learning.
        
        Reference: fpsan_proofsheet.py L144-L148
        """
        if rate <= 0:
            return
        self.stiffness.add_(self.daily_activity * rate).clamp_(max=self.stiffness_cap)
        self.daily_activity.zero_()

    @torch.no_grad()
    def get_ternary_weights(self) -> torch.Tensor:
        """Export pure ternary {-1, 0, +1} int8 weights for C++/HLSL inference."""
        scale = self.weight.abs().mean().clamp_min(1e-5)
        return torch.round(self.weight / scale).clamp(-1, 1).to(torch.int8)

    @torch.no_grad()
    def get_weight_scale(self) -> float:
        """Get the per-layer scale factor for ternary reconstruction."""
        return self.weight.abs().mean().clamp_min(1e-5).item()

    def get_sparsity(self) -> float:
        """Return the fraction of zero (inactive) ternary weights."""
        w_ternary = self.get_ternary_weights()
        return (w_ternary == 0).float().mean().item()

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, groups={self.groups}"
        )
        sparsity = self.get_sparsity() * 100
        return s + f", sparsity={sparsity:.1f}%"


class FP16Conv2d(nn.Module):
    """
    Standard FP16 convolution for the color-precision-critical reconstruction head.
    
    Used only for the PixelShuffle reconstruction layer (Decision 3: Hybrid Precision).
    Everything else uses FPSANConv2d (ternary).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use FP16 for precision-critical color reconstruction
        if x.is_cuda and x.dtype == torch.float32:
            with torch.amp.autocast("cuda"):
                return self.conv(x)
        return self.conv(x)
