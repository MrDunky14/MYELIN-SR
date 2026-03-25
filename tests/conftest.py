"""Pytest configuration and fixtures for FP-SAN NSS tests."""

import pytest
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))


@pytest.fixture
def device():
    """Provide device for tests (CPU or CUDA if available)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_lr_image():
    """Provide a dummy low-resolution image."""
    import torch
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def dummy_hr_image():
    """Provide a dummy high-resolution image."""
    import torch
    return torch.randn(1, 3, 128, 128)
