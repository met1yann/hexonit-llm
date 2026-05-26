"""
Quantization Advisor — hexonit-llm's killer feature.
Analyzes hardware and recommends the optimal quantization BEFORE downloading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Approximate VRAM requirements per quantization level (GB per billion params)
_VRAM_PER_BILLION: dict[str, float] = {
    "fp16":    2.0,
    "q8_0":   1.0,
    "q6_k":   0.75,
    "q5_k_m": 0.625,
    "q4_k_m": 0.5,
    "q3_k_m": 0.375,
    "q2_k":   0.25,
}

# Known model parameter counts (billions)
_MODEL_PARAMS: dict[str, float] = {
    "meta-llama/meta-llama-3-70b-instruct":   70,
    "meta-llama/meta-llama-3-8b-instruct":     8,
    "qwen/qwen2.5-72b-instruct":              72,
    "qwen/qwen2.5-7b-instruct":                7,
    "mistralai/mixtral-8x22b-instruct":       39,
    "google/gemma-2-27b-it":                  27,
    "google/gemma-2-9b-it":                    9,
    "deepseek-ai/deepseek-v2.5":              16,
    "microsoft/phi-3-medium-4k-instruct":     14,
    "microsoft/phi-3-mini-4k-instruct":        3.8,
}


@dataclass
class QuantizationAdvice:
    """Result of a quantization compatibility check."""

    can_run: bool
    recommended_quant: str
    estimated_vram_gb: float
    available_vram_gb: float
    deficit_gb: float
    surplus_gb: float
    fallback_suggestion: str | None
    explanation: str

    def __str__(self) -> str:
        if self.can_run:
            surplus_pct = (self.surplus_gb / self.available_vram_gb) * 100 if self.available_vram_gb else 0
            return (
                f"✅ Can run | Recommended: {self.recommended_quant.upper()} | "
                f"Est. VRAM: {self.estimated_vram_gb:.1f}GB / {self.available_vram_gb:.1f}GB available "
                f"({surplus_pct:.0f}% headroom)\n"
                f"   {self.explanation}"
            )
        return (
            f"❌ Cannot run | Need {self.estimated_vram_gb:.1f}GB, have {self.available_vram_gb:.1f}GB "
            f"(deficit: {self.deficit_gb:.1f}GB)\n"
            f"   {self.explanation}\n"
            + (f"   💡 Try instead: {self.fallback_suggestion}" if self.fallback_suggestion else "")
        )


def _estimate_params_b(model_name: str) -> float:
    """Estimate parameter count from model name. Returns default 7B if unknown."""
    key = model_name.lower()

    # Direct lookup
    if key in _MODEL_PARAMS:
        return _MODEL_PARAMS[key]

    # Partial match
    for k, v in _MODEL_PARAMS.items():
        k_short = k.split("/")[-1]
        if k_short in key or key.split("/")[-1] in k_short:
            return v

    # Heuristic: extract number before 'b' (e.g., "70b" → 70, "1.5b" → 1.5)
    match = re.search(r"(\d+\.?\d*)\s*b", key.replace("-", " ").replace("_", " "))
    if match:
        return float(match.group(1))

    # Try to find raw digits followed by b
    match = re.search(r"(\d+\.?\d*)b", key)
    if match:
        return float(match.group(1))

    return 7.0  # safe default


def _suggest_smaller_model(params_b: float, available_vram_gb: float) -> str | None:
    """Suggest a smaller model that fits in available VRAM."""
    q4_per_b = _VRAM_PER_BILLION["q4_k_m"]
    max_params = (available_vram_gb / 1.1) / q4_per_b

    suggestions = {
        70: "meta-llama/Meta-Llama-3-70B-Instruct",
        8:  "meta-llama/Meta-Llama-3-8B-Instruct",
        7:  "Qwen/Qwen2.5-7B-Instruct",
        3:  "microsoft/Phi-3-mini-4k-instruct",
        1:  "Qwen/Qwen2.5-1.5B-Instruct",
    }

    for size, name in sorted(suggestions.items(), reverse=True):
        if size <= max_params:
            return f"{name} ({size}B) fits at Q4_K_M"
    return None


def advise(model_name: str, available_vram_gb: float) -> QuantizationAdvice:
    """
    Given a model name and available VRAM, recommend the best quantization level.

    Args:
        model_name: HuggingFace model ID (case-insensitive)
        available_vram_gb: Available VRAM in gigabytes (0.0 if no GPU)

    Returns:
        QuantizationAdvice with full recommendation
    """
    params_b = _estimate_params_b(model_name)

    # Handle no GPU case
    if available_vram_gb <= 0:
        return QuantizationAdvice(
            can_run=False,
            recommended_quant="q2_k",
            estimated_vram_gb=params_b * _VRAM_PER_BILLION["q2_k"] * 1.1,
            available_vram_gb=0.0,
            deficit_gb=params_b * _VRAM_PER_BILLION["q2_k"] * 1.1,
            surplus_gb=0.0,
            fallback_suggestion="No GPU detected. Try llama.cpp with CPU offloading.",
            explanation=f"No GPU VRAM available. {params_b}B model requires a GPU with at least "
                        f"{params_b * _VRAM_PER_BILLION['q2_k'] * 1.1:.1f}GB VRAM.",
        )

    # Find best quant that fits
    recommended = None
    estimated_vram = None
    for quant, vram_per_b in _VRAM_PER_BILLION.items():
        needed = params_b * vram_per_b * 1.1  # +10% overhead for KV cache
        if needed <= available_vram_gb:
            recommended = quant
            estimated_vram = needed
            break

    if recommended is None:
        # Can't run even at lowest quant
        lowest = "q2_k"
        min_needed = params_b * _VRAM_PER_BILLION[lowest] * 1.1
        fallback = _suggest_smaller_model(params_b, available_vram_gb)
        return QuantizationAdvice(
            can_run=False,
            recommended_quant=lowest,
            estimated_vram_gb=min_needed,
            available_vram_gb=available_vram_gb,
            deficit_gb=min_needed - available_vram_gb,
            surplus_gb=0.0,
            fallback_suggestion=fallback,
            explanation=(
                f"Model requires minimum {min_needed:.1f}GB even at Q2_K quantization. "
                f"Your {available_vram_gb:.1f}GB VRAM is insufficient."
            ),
        )

    surplus = available_vram_gb - estimated_vram
    return QuantizationAdvice(
        can_run=True,
        recommended_quant=recommended,
        estimated_vram_gb=estimated_vram,
        available_vram_gb=available_vram_gb,
        deficit_gb=0.0,
        surplus_gb=surplus,
        fallback_suggestion=None,
        explanation=(
            f"{params_b:.0f}B parameter model at {recommended.upper()} uses ~{estimated_vram:.1f}GB "
            f"including KV cache overhead."
        ),
    )