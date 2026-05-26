"""
Hardware & Operating-System detection module.

Every public function in this module is designed to be **safe to import on any
platform** – no backend-specific imports are made at module level.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass
class HardwareProfile:
    """Immutable snapshot of the current system's hardware capabilities."""

    # Operating system
    os_type: str = ""  # "Linux" | "Windows" | "Darwin" (macOS)
    os_version: str = ""

    # GPU
    cuda_available: bool = False
    cuda_device_count: int = 0
    total_vram_gb: float = 0.0   # aggregate VRAM across all devices (GB)
    free_vram_gb: float = 0.0    # aggregate free VRAM (GB)
    gpu_names: list[str] = field(default_factory=list)

    # System RAM
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0

    # CPU
    cpu_count: int = 0
    cpu_model: str = ""


def _probe_cuda() -> tuple[bool, int, float, float, list[str]]:
    """
    Attempt to import ``torch`` and query CUDA information.

    Returns
    -------
    (available, device_count, total_vram_gb, free_vram_gb, device_names)
    """
    available = False
    device_count = 0
    total_vram = 0.0
    free_vram = 0.0
    names: list[str] = []

    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_vram += props.total_memory
                free_vram += props.total_memory - torch.cuda.memory_reserved(i)
                names.append(props.name)
            # Convert bytes → GB
            total_vram = total_vram / (1024 ** 3)
            free_vram = free_vram / (1024 ** 3)
    except (ImportError, RuntimeError, Exception):
        pass  # torch not installed or no CUDA

    return available, device_count, total_vram, free_vram, names


def _probe_ram() -> tuple[float, float]:
    """Return (total_ram_gb, available_ram_gb)."""
    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3), mem.available / (1024 ** 3)


def _probe_cpu() -> tuple[int, str]:
    """Return (logical_core_count, model_name_string)."""
    count = psutil.cpu_count(logical=True) or 0
    model = platform.processor() or "unknown"
    return count, model


def detect_hardware() -> HardwareProfile:
    """
    Perform a one-shot detection of the current machine's hardware.

    This function is **safe to call on any OS** – CUDA-related errors are
    silently caught and the corresponding fields will remain at their defaults.

    Returns
    -------
    HardwareProfile
        A frozen dataclass holding all detected specs.
    """
    os_type = platform.system()
    os_version = platform.release()

    cuda_avail, dev_count, total_vram, free_vram, gpu_names = _probe_cuda()
    total_ram, avail_ram = _probe_ram()
    cpu_count, cpu_model = _probe_cpu()

    return HardwareProfile(
        os_type=os_type,
        os_version=os_version,
        cuda_available=cuda_avail,
        cuda_device_count=dev_count,
        total_vram_gb=round(total_vram, 2),
        free_vram_gb=round(free_vram, 2),
        gpu_names=gpu_names,
        total_ram_gb=round(total_ram, 2),
        available_ram_gb=round(avail_ram, 2),
        cpu_count=cpu_count,
        cpu_model=cpu_model,
    )


# ── Public routing queries ──────────────────────────────────

def is_linux() -> bool:
    """Return ``True`` when the OS is Linux."""
    return platform.system() == "Linux"


def is_windows() -> bool:
    """Return ``True`` when the OS is Windows."""
    return platform.system() == "Windows"


def is_macos() -> bool:
    """Return ``True`` when the OS is macOS (Darwin)."""
    return platform.system() == "Darwin"


def can_use_vllm(hw: Optional[HardwareProfile] = None) -> bool:
    """
    Determine whether the system is suitable for the **vLLM** engine.

    Rules
    -----
    * OS must be **Linux**.
    * Aggregated VRAM must be **≥ 16 GB**.
    * ``torch.cuda`` must be available.
    """
    if hw is None:
        hw = detect_hardware()

    return (
        hw.os_type == "Linux"
        and hw.cuda_available
        and hw.total_vram_gb >= 16.0
    )


def can_use_llamacpp(hw: Optional[HardwareProfile] = None) -> bool:
    """
    Determine whether the **llama.cpp** engine should be used.

    Rules
    -----
    * Use on **Windows** or **macOS** regardless of VRAM.
    * Also use on **Linux** when VRAM < 16 GB (fallback).
    """
    if hw is None:
        hw = detect_hardware()

    return hw.os_type in ("Windows", "Darwin") or (
        hw.os_type == "Linux" and hw.total_vram_gb < 16.0
    )


def select_engine() -> str:
    """
    High-level router: returns ``"vllm"`` or ``"llamacpp"`` based on current
    hardware inspection.  This is the single entry-point used by the
    orchestrator.
    """
    if can_use_vllm():
        return "vllm"
    return "llamacpp"