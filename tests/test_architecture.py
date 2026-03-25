"""Tests for MYELIN-SR architecture components and forward/backward passes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

import torch
import pytest
from model import build_myelin_sr
from fpsan_conv2d import FPSANConv2d


class TestArchitectureComprehensive:
    """Comprehensive architecture validation tests."""

    def test_all_presets_forward_pass(self):
        """Test forward pass for all quality presets."""
        presets = ['performance', 'balanced', 'quality', 'ultra']
        for preset in presets:
            model = build_myelin_sr(2, preset)
            dummy_input = torch.randn(1, 3, 64, 64)
            output = model(dummy_input)
            assert output.shape == (1, 3, 128, 128), f"Preset {preset} failed shape check"
            assert not torch.isnan(output).any(), f"Preset {preset} produced NaN"
            assert not torch.isinf(output).any(), f"Preset {preset} produced Inf"

    def test_model_parameter_counts(self):
        """Test that parameter counts match expected values per preset."""
        expected_counts = {
            'performance': 8742,
            'balanced': 39320,
            'quality': 39488,
            'ultra': 92256,
        }
        for preset, expected in expected_counts.items():
            model = build_myelin_sr(2, preset)
            actual = sum(p.numel() for p in model.parameters())
            assert actual == expected, f"Preset {preset}: expected {expected}, got {actual}"

    def test_fpsan_layer_count(self):
        """Test that model contains expected number of FPSANConv2d layers."""
        model = build_myelin_sr(2, 'quality')
        fpsan_count = sum(1 for m in model.modules() if isinstance(m, FPSANConv2d))
        assert fpsan_count == 21, f"Expected 21 FPSANConv2d layers, got {fpsan_count}"

    def test_backward_pass_gradient_flow(self):
        """Test that gradients flow through all trainable parameters."""
        model = build_myelin_sr(2, 'quality')
        model.train()
        
        dummy_lr = torch.randn(1, 3, 64, 64)
        dummy_hr = torch.randn(1, 3, 128, 128)
        
        sr = model(dummy_lr)
        loss = torch.nn.functional.l1_loss(sr, dummy_hr)
        loss.backward()
        
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        
        assert params_with_grad == total_trainable, f"Gradient flow incomplete: {params_with_grad}/{total_trainable}"

    def test_model_deploy_size(self):
        """Test that deploy size calculation works."""
        model = build_myelin_sr(2, 'quality')
        deploy_size = model.get_deploy_size_bytes()
        assert deploy_size > 0, "Deploy size should be positive"
        # Should be less than 100KB
        assert deploy_size < 100 * 1024, f"Deploy size too large: {deploy_size} bytes"

    def test_model_sparsity(self):
        """Test that ternary sparsity is within expected range."""
        model = build_myelin_sr(2, 'quality')
        sparsity = model.get_total_sparsity()
        assert 0.15 < sparsity < 0.35, f"Sparsity out of range: {sparsity}"


class TestTernaryQuantization:
    """Tests for ternary weight quantization."""

    def test_ternary_weight_values(self):
        """Test that ternary weights are in {-1, 0, 1}."""
        model = build_myelin_sr(2, 'quality')
        for module in model.modules():
            if isinstance(module, FPSANConv2d):
                ternary_weights = module.get_ternary_weights()
                unique_vals = torch.unique(ternary_weights).tolist()
                for val in unique_vals:
                    assert val in [-1, 0, 1], f"Invalid ternary value: {val}"

    def test_weight_scale_factors(self):
        """Test that weight scale factors are positive."""
        model = build_myelin_sr(2, 'quality')
        for module in model.modules():
            if isinstance(module, FPSANConv2d):
                scale = module.get_weight_scale()
                assert scale > 0, f"Scale factor should be positive, got {scale}"


class TestConsolidation:
    """Tests for sleep consolidation mechanism."""

    def test_sleep_consolidation_preserves_shape(self):
        """Test that consolidation preserves model structure."""
        model = build_myelin_sr(2, 'quality')
        original_param_count = sum(p.numel() for p in model.parameters())
        
        model.consolidate_all(rate=0.08)
        
        new_param_count = sum(p.numel() for p in model.parameters())
        assert original_param_count == new_param_count, "Consolidation changed parameter count"

    def test_consolidation_completes(self):
        """Test that consolidation runs without errors."""
        model = build_myelin_sr(2, 'quality')
        try:
            model.consolidate_all(rate=0.08)
            assert True
        except Exception as e:
            assert False, f"Consolidation failed: {e}"
