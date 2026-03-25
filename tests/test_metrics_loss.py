"""Tests for metrics (PSNR, SSIM) and loss functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

import torch
import pytest
from metrics import calculate_psnr, calculate_ssim, rgb_to_ycbcr
from losses import MyelinSRLoss


class TestMetricFunctions:
    """Tests for PSNR and SSIM metric calculations."""

    def test_psnr_identical_images(self):
        """Test PSNR returns ~100 dB for identical images."""
        img = torch.rand(1, 3, 64, 64)
        psnr = calculate_psnr(img, img)
        assert psnr > 50, f"PSNR for identical images should be high, got {psnr}"

    def test_psnr_reasonable_noisy_images(self):
        """Test PSNR is reasonable for noisy images."""
        img1 = torch.rand(1, 3, 64, 64)
        img2 = img1 + torch.randn_like(img1) * 0.1
        img2 = img2.clamp(0, 1)
        psnr = calculate_psnr(img1, img2)
        assert 15 < psnr < 40, f"PSNR for noisy images out of range: {psnr}"

    def test_ssim_identical_images(self):
        """Test SSIM returns ~1.0 for identical images."""
        img = torch.rand(1, 3, 64, 64)
        ssim = calculate_ssim(img, img)
        assert ssim > 0.99, f"SSIM for identical images should be ~1.0, got {ssim}"

    def test_ssim_different_images(self):
        """Test SSIM is reasonable for different images."""
        img1 = torch.zeros(1, 3, 64, 64)
        img2 = torch.ones(1, 3, 64, 64)
        ssim = calculate_ssim(img1, img2)
        assert 0.0 <= ssim <= 1.0, f"SSIM out of valid range: {ssim}"

    def test_rgb_to_ycbcr_conversion(self):
        """Test RGB to YCbCr conversion produces valid output."""
        rgb = torch.rand(2, 3, 64, 64)
        y = rgb_to_ycbcr(rgb)
        assert y.shape == (2, 1, 64, 64), "YCbCr conversion changed tensor shape"
        assert y.min() >= 0, "YCbCr Y channel should be non-negative"
        assert y.max() <= 1, "YCbCr Y channel should be <= 1"


class TestLossFunction:
    """Tests for MyelinSRLoss."""

    def test_loss_forward_pass(self):
        """Test loss function forward pass."""
        criterion = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False, use_freq=False)
        sr = torch.randn(2, 3, 64, 64, requires_grad=True)
        hr = torch.randn(2, 3, 64, 64)
        
        total_loss, loss_dict = criterion(sr, hr)
        assert isinstance(total_loss, torch.Tensor), "Loss should be a tensor"
        assert 'charbonnier' in loss_dict, "Charbonnier loss should be in loss dict"
        assert 'total' in loss_dict, "Total loss should be in loss dict"

    def test_loss_backward_pass(self):
        """Test that loss can be backpropagated."""
        criterion = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False)
        sr = torch.randn(2, 3, 64, 64, requires_grad=True)
        hr = torch.randn(2, 3, 64, 64)
        
        total_loss, loss_dict = criterion(sr, hr)
        try:
            total_loss.backward()
            assert sr.grad is not None, "Gradients should flow to SR output"
        except Exception as e:
            assert False, f"Loss backward pass failed: {e}"

    def test_loss_pixel_component(self):
        """Test Charbonnier loss component is reasonable."""
        criterion = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False, use_freq=False)
        sr = torch.zeros(2, 3, 64, 64)
        hr = torch.ones(2, 3, 64, 64)
        
        total_loss, loss_dict = criterion(sr, hr)
        pixel_loss = loss_dict['charbonnier']
        assert pixel_loss > 0, "Charbonnier loss should be positive for different images"
        assert pixel_loss < 2.0, "Charbonnier loss should be reasonable for [0,1] range"

    def test_loss_identical_images(self):
        """Test loss is very small for identical images (Charbonnier floor = epsilon = 1e-3)."""
        criterion = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False, use_freq=False)
        img = torch.rand(2, 3, 64, 64)
        
        total_loss, loss_dict = criterion(img, img)
        # Charbonnier at zero diff = epsilon = 1e-3, not exactly 0
        assert total_loss < 5e-3, f"Loss for identical images should be near epsilon, got {total_loss}"
