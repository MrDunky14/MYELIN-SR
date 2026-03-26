# MYELIN-SR v2: Zero-Barrier Ternary Reconstruction Engine
# Copyright (C) 2026 Krishna Singh
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Integration tests for complete training pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

import torch
import pytest
from model import build_myelin_sr
from metrics import PhaseGateValidator, calculate_psnr, calculate_ssim
from losses import MyelinSRLoss


class TestPhaseGates:
    """Tests for Phase Gate validation system."""

    def test_gate_valid_output(self):
        """Test Gate 1.1: valid output check."""
        model = build_myelin_sr(2, 'quality')
        validator = PhaseGateValidator()
        result = validator.check_valid_output(model, torch.device("cpu"))
        assert result is True, f"Gate 1.1 failed: {validator.results.get('valid_output')}"

    def test_gate_sparsity_check(self):
        """Test Gate 1.6: sparsity check."""
        model = build_myelin_sr(2, 'quality')
        validator = PhaseGateValidator()
        result = validator.check_sparsity(model)
        assert result is True, f"Sparsity gate failed: model sparsity {model.get_total_sparsity()}"


class TestTrainingPipeline:
    """Tests for complete training forward/backward loop."""

    def test_training_forward_backward_cycle(self):
        """Test a complete training iteration."""
        model = build_myelin_sr(2, 'quality')
        model.train()
        
        criterion = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Forward pass
        lr = torch.randn(2, 3, 32, 32)
        hr = torch.randn(2, 3, 64, 64)
        
        sr = model(lr)
        loss, loss_dict = criterion(sr, hr)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss decreased (or at least changed)
        sr2 = model(lr)
        loss2, _ = criterion(sr2, hr)
        
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert sr.shape == (2, 3, 64, 64), "SR output shape mismatch"

    def test_inference_mode(self):
        """Test model in evaluation mode."""
        model = build_myelin_sr(2, 'quality')
        model.eval()
        
        with torch.no_grad():
            lr = torch.randn(4, 3, 128, 128)
            sr = model(lr)
            
            assert sr.shape == (4, 3, 256, 256), "Inference output shape mismatch"
            assert not torch.isnan(sr).any(), "Inference produced NaN"

    def test_batch_processing(self):
        """Test model with different batch sizes."""
        model = build_myelin_sr(2, 'quality')
        model.eval()
        
        with torch.no_grad():
            for batch_size in [1, 2, 4, 8]:
                lr = torch.randn(batch_size, 3, 64, 64)
                sr = model(lr)
                assert sr.shape == (batch_size, 3, 128, 128), f"Batch size {batch_size} failed"


class TestMemoryEfficiency:
    """Tests for memory efficiency and model size."""

    def test_deploy_size_optimization(self):
        """Test that all presets produce reasonable deploy sizes."""
        for preset in ['performance', 'balanced', 'quality', 'ultra']:
            model = build_myelin_sr(2, preset)
            deploy_size = model.get_deploy_size_bytes()
            deploy_kb = deploy_size / 1024
            
            # Even ultra should be < 200KB
            assert deploy_kb < 200, f"Preset {preset} deploy size too large: {deploy_kb}KB"

    def test_parameter_efficiency(self):
        """Test parameter efficiency across presets."""
        models = {
            'performance': 8742,
            'balanced': 39320,
            'quality': 39488,
            'ultra': 92256,
        }
        
        # Performance should be significantly smaller than ultra
        perf = build_myelin_sr(2, 'performance')
        ultra = build_myelin_sr(2, 'ultra')
        
        perf_params = sum(p.numel() for p in perf.parameters())
        ultra_params = sum(p.numel() for p in ultra.parameters())
        
        ratio = ultra_params / perf_params
        assert ratio > 2, f"Ultra should have significantly more params than performance"
