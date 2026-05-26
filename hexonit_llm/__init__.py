"""
hexonit-llm — Ultra-fast local LLM inference, zero config.

Single-import, zero-configuration library for running large language models
locally at maximum speed.  Automatically detects your hardware, selects the
optimal inference engine (vLLM or llama.cpp), and enables speculative decoding
with the best matching draft model.

Three operating modes:

**Mode 1 – Cloud-Assisted Draft (Hybrid)**
    Provide ``draft_base_url`` + ``api_key``.
    → Target model loads locally; draft tokens come from ANY OpenAI-compatible
      endpoint (Groq, OpenRouter, Together, SambaNova, local vLLM, etc.).
    → Falls back gracefully to single-model local if network fails.

**Mode 2 – Explicit Local Pair (max speed)**
    Provide ``draft_model`` explicitly, leave ``draft_base_url=None``.
    → BOTH models load locally for speculative decoding.

**Mode 3 – Auto Local (default, zero config)**
    Provide only ``model``.
    → Best draft auto-selected from built-in mapping (optional suggestion).
    → If no mapping found, runs single-model inference seamlessly.

Typical usage::

    from hexonit_llm import UltraInference

    # Auto local (zero config)
    pipe = UltraInference("meta-llama/Meta-Llama-3-70B-Instruct")
    response = pipe.generate("What is the meaning of life?")
    print(response)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hexonit_llm.orchestrator import Orchestrator

# ── Package metadata ─────────────────────────────────────────
__version__ = "0.0.2"
__author__ = "Hexonithy Studios"
__all__ = ["UltraInference"]

# ── Logging ──────────────────────────────────────────────────
_logger = logging.getLogger("hexonit_llm")
_logger.addHandler(logging.NullHandler())


def _setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple stderr handler if none is configured."""
    if not _logger.handlers or all(
        isinstance(h, logging.NullHandler) for h in _logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[hexonit-llm] %(message)s")
        )
        _logger.addHandler(handler)
    _logger.setLevel(level)


# ── Public API ───────────────────────────────────────────────

class UltraInference:
    """
    Zero-config local LLM inference with automatic hardware routing,
    speculative decoding, and optional cloud-assisted drafting.

    This is the **only** class the end-user ever needs to import.

    Parameters
    ----------
    model : str
        ANY Hugging Face model ID or local path.
        Examples: ``"meta-llama/Meta-Llama-3-70B-Instruct"``,
        ``"Qwen/Qwen2.5-72B-Instruct"``,
        ``"/models/my-model.gguf"``.
        No model is blocked. Built-in mappings are suggestions only.
    draft_model : str, optional
        Explicit local draft model for speculative decoding (Mode 2).
        When provided with ``draft_base_url=None``, BOTH models run locally.
    draft_base_url : str, optional
        Base URL of any OpenAI-compatible chat completions API (Mode 1).
        Examples:
        - ``"https://api.groq.com/openai/v1"``
        - ``"https://openrouter.ai/api/v1"``
        - ``"http://localhost:8000/v1"``
        Requires ``api_key``.
    api_key : str, optional
        API key for the remote endpoint.
    optimization_level : str, optional
        One of ``"max_speed"``, ``"balanced"``, ``"memory_saver"``.
        Default is ``"max_speed"``.
    verbose : bool, optional
        Enable detailed logging output. Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to the underlying engine.

    Examples
    --------
    >>> from hexonit_llm import UltraInference

    # Auto local (default — any model works)
    >>> pipe = UltraInference("Qwen/Qwen2.5-72B-Instruct")
    >>> print(pipe.generate("Tell me a joke"))
    """

    def __init__(
        self,
        model: str,
        draft_model: Optional[str] = None,
        draft_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        optimization_level: str = "max_speed",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        if verbose:
            _setup_logging(logging.DEBUG)
        else:
            _setup_logging(logging.WARNING)

        self._orchestrator = Orchestrator(
            model=model,
            optimization_level=optimization_level,
            explicit_draft_model=draft_model,
            draft_base_url=draft_base_url,
            api_key=api_key,
            **kwargs,
        )

    # ── Properties ───────────────────────────────────────────

    @property
    def engine_name(self) -> str:
        """Name of the engine in use (``"vllm"`` or ``"llamacpp"``)."""
        return self._orchestrator.engine_name

    @property
    def draft_model(self) -> Optional[str]:
        """Draft model ID used for speculative decoding, or ``None``."""
        return self._orchestrator.draft_model

    @property
    def mode(self) -> str:
        """
        Current operating mode.

        One of: ``"hybrid_cloud"``, ``"local_explicit_pair"``,
        ``"local_auto"``, ``"local_single"``.
        """
        return self._orchestrator.mode

    @property
    def hardware_info(self) -> dict[str, Any]:
        """Summary of detected hardware as a dictionary."""
        hw = self._orchestrator.hw
        return {
            "os": hw.os_type,
            "cuda_available": hw.cuda_available,
            "total_vram_gb": hw.total_vram_gb,
            "total_ram_gb": hw.total_ram_gb,
            "cpu_count": hw.cpu_count,
            "gpu_names": hw.gpu_names,
        }

    # ── Generation ───────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        **sampling_kwargs: Any,
    ) -> str:
        """
        Generate a completion for a single prompt.

        Parameters
        ----------
        prompt : str
            Input text.
        **sampling_kwargs
            Optional overrides:
            ``temperature`` (float), ``top_p`` (float), ``max_tokens`` (int).

        Returns
        -------
        str
            Generated text.
        """
        return self._orchestrator.generate(prompt, **sampling_kwargs)

    def generate_batch(
        self,
        prompts: list[str],
        **sampling_kwargs: Any,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.

        Parameters
        ----------
        prompts : list[str]
            Multiple input texts.
        **sampling_kwargs
            Optional overrides.

        Returns
        -------
        list[str]
            Generated texts, one per input.
        """
        return self._orchestrator.generate_batch(prompts, **sampling_kwargs)

    # ── Chat ─────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        **sampling_kwargs: Any,
    ) -> str:
        """
        OpenAI-compatible chat completion.

        Parameters
        ----------
        messages : list[dict]
            Example::

                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": "What is 2+2?"}]

        **sampling_kwargs
            Optional overrides.

        Returns
        -------
        str
            Assistant reply.
        """
        return self._orchestrator.chat(messages, **sampling_kwargs)

    def __repr__(self) -> str:
        return (
            f"UltraInference(model={self._orchestrator.model!r}, "
            f"engine={self.engine_name!r}, "
            f"mode={self.mode!r})"
        )