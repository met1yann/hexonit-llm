"""
vLLM engine wrapper.

**IMPORTANT:** This module MUST NOT be imported on non-Linux platforms.
All top-level imports of ``vllm`` are performed lazily inside functions so
that simply loading the module on Windows/macOS does not crash the process.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Optional

from hexonit_llm.engines.cloud_draft_provider import CloudDraftClient

logger = logging.getLogger("hexonit_llm.engines.vllm")


# ── Error raised when vLLM is not available ─────────────────
class VllmUnavailableError(RuntimeError):
    """Raised when vLLM cannot be loaded (wrong platform or missing package)."""


# ── Hardcoded optimisation presets ──────────────────────────

@dataclass
class VllmConfig:
    """
    Frozen configuration that will be passed verbatim to ``vllm.LLM``.

    These values are chosen for **maximum token throughput** and are not
    user-exposed (zero-config philosophy).
    """

    # Speculative decoding
    draft_model: Optional[str] = None

    # Memory & scheduling
    gpu_memory_utilization: float = 0.95
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True

    # Attention / compute
    enable_flash_attention: bool = True      # FlashAttention-2
    enforce_eager: bool = False              # use CUDA graphs

    # Prefix caching boosts throughput for repeated prefixes
    enable_prefix_caching: bool = True

    # Misc
    trust_remote_code: bool = True
    dtype: str = "auto"                      # bfloat16 if available, else float16
    seed: int = 42

    # Extra kwargs forwarded to vLLM
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# ── Lazy vLLM loader ────────────────────────────────────────

_VLLM_AVAILABLE: bool | None = None  # cache


def _check_vllm_available() -> bool:
    """
    Return ``True`` iff we are on **Linux** and **vllm** can be imported.
    """
    global _VLLM_AVAILABLE
    if _VLLM_AVAILABLE is not None:
        return _VLLM_AVAILABLE

    if platform.system() != "Linux":
        logger.debug("vLLM is unavailable: not on Linux (detected %s)", platform.system())
        _VLLM_AVAILABLE = False
        return False

    try:
        import vllm  # noqa: F401
        _VLLM_AVAILABLE = True
    except ImportError:
        logger.warning("vLLM package is not installed. Install with: pip install hexonit-llm[vllm]")
        _VLLM_AVAILABLE = False
    except Exception as exc:
        logger.warning("vLLM import failed: %s", exc)
        _VLLM_AVAILABLE = False

    return _VLLM_AVAILABLE


# ── Public engine class ─────────────────────────────────────

class VllmEngine:
    """
    Thin wrapper around ``vllm.LLM`` that pre‑configures speculative decoding
    and high‑throughput settings.

    Usage
    -----
    .. code-block:: python

        engine = VllmEngine("meta-llama/Meta-Llama-3-70B-Instruct")
        outputs = engine.generate(["Your prompt here"])
    """

    def __init__(
        self,
        model: str,
        draft_model: Optional[str] = None,
        cloud_draft_client: Optional[CloudDraftClient] = None,
    ) -> None:
        if not _check_vllm_available():
            raise VllmUnavailableError(
                "vLLM is not available on this system. "
                "It requires Linux with CUDA and the vllm Python package."
            )

        self.model_name = model
        self._config = VllmConfig(draft_model=draft_model)
        self._cloud_draft_client = cloud_draft_client
        self._llm: Any = None  # will hold vllm.LLM instance

    # ── Lazy initialisation ──────────────────────────────────

    def _init_llm(self) -> None:
        """Actually instantiate ``vllm.LLM`` (called on first ``generate``)."""
        if self._llm is not None:
            return

        from vllm import LLM, SamplingParams  # lazy import

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "gpu_memory_utilization": self._config.gpu_memory_utilization,
            "max_num_seqs": self._config.max_num_seqs,
            "max_num_batched_tokens": self._config.max_num_batched_tokens,
            "enable_chunked_prefill": self._config.enable_chunked_prefill,
            "enable_prefix_caching": self._config.enable_prefix_caching,
            "trust_remote_code": self._config.trust_remote_code,
            "dtype": self._config.dtype,
            "seed": self._config.seed,
        }

        # Speculative decoding
        if self._config.draft_model is not None:
            kwargs["draft_model"] = self._config.draft_model

        # Flash attention
        if self._config.enable_flash_attention:
            kwargs["enable_flash_attention"] = True

        # CUDA graphs
        kwargs["enforce_eager"] = self._config.enforce_eager

        # Merge extra kwargs (user-level overrides)
        kwargs.update(self._config.extra_kwargs)

        logger.info(
            "Initialising vLLM engine (model=%s, draft=%s)",
            self.model_name,
            self._config.draft_model or "none",
        )
        self._llm = LLM(**kwargs)

        # Store a default sampling params object
        self._sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
        )

    # ── Public generate method ───────────────────────────────

    def generate(
        self,
        prompts: str | list[str],
        **sampling_kwargs: Any,
    ) -> list[str]:
        """
        Run inference.

        Parameters
        ----------
        prompts : str | list[str]
            One or more prompts.
        **sampling_kwargs
            Overrides for ``SamplingParams`` (e.g. ``temperature=0.1``).

        Returns
        -------
        list[str]
            Generated text for each prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        self._init_llm()

        from vllm import SamplingParams

        params = SamplingParams(
            temperature=sampling_kwargs.get("temperature", 0.7),
            top_p=sampling_kwargs.get("top_p", 0.9),
            max_tokens=sampling_kwargs.get("max_tokens", 2048),
        )

        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]