"""Shared pytest fixtures for hexonit-llm tests."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from hexonit_llm.utils.hardware_detector import HardwareProfile


@pytest.fixture
def mock_hw_8gb() -> HardwareProfile:
    """Linux with 8GB VRAM (should route to llama.cpp)."""
    return HardwareProfile(
        os_type="Linux",
        total_vram_gb=8.0,
        total_ram_gb=32.0,
        cpu_count=16,
        cuda_available=True,
        cuda_device_count=1,
        gpu_names=["NVIDIA RTX 3070"],
    )


@pytest.fixture
def mock_hw_24gb() -> HardwareProfile:
    """Linux with 24GB VRAM (should route to vLLM)."""
    return HardwareProfile(
        os_type="Linux",
        total_vram_gb=24.0,
        total_ram_gb=64.0,
        cpu_count=32,
        cuda_available=True,
        cuda_device_count=1,
        gpu_names=["NVIDIA RTX 4090"],
    )


@pytest.fixture
def mock_hw_no_gpu() -> HardwareProfile:
    """Windows with no GPU (should route to llama.cpp)."""
    return HardwareProfile(
        os_type="Windows",
        total_vram_gb=0.0,
        total_ram_gb=16.0,
        cpu_count=8,
        cuda_available=False,
    )