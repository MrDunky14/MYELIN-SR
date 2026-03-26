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
