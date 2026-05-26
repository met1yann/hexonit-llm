"""
llama.cpp engine wrapper (using ``llama-cpp-python``).

Works on **Windows**, **macOS** (Metal backend), and **Linux** when VRAM is
limited.  Handles GGUF model downloading / conversion transparently.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hexonit_llm.engines.cloud_draft_provider import CloudDraftClient

logger = logging.getLogger("hexonit_llm.engines.llamacpp")


# ── Error class ──────────────────────────────────────────────
class LlamaCppUnavailableError(RuntimeError):
    """Raised when llama-cpp-python is not installed."""


# ── Hardcoded optimisation presets ──────────────────────────

@dataclass
class LlamaCppConfig:
    """
    Pre-tuned configuration for maximum token throughput on llama.cpp.

    All values are zero-config – the user never touches these.
    """

    # GPU offloading: -1 means "all layers"
    n_gpu_layers: int = -1

    # Context size
    n_ctx: int = 4096
    n_batch: int = 2048
    n_ubatch: int = 512
    n_threads: int = os.cpu_count() or 4

    # Draft model for speculative decoding
    draft_model: Optional[str] = None
    draft_model_path: Optional[Path] = None

    # Attention & compute
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False  # mlock can conflict with offloading

    # Quantisation: load in original quant; the user's GGUF file
    # already determines the type.
    model_format: str = "gguf"

    # Verbose logging
    verbose: bool = False

    # Extra kwargs
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# ── Lazy loader ─────────────────────────────────────────────

_LLAMACPP_AVAILABLE: bool | None = None


def _check_llamacpp_available() -> bool:
    """Return ``True`` iff ``llama_cpp`` can be imported."""
    global _LLAMACPP_AVAILABLE
    if _LLAMACPP_AVAILABLE is not None:
        return _LLAMACPP_AVAILABLE

    try:
        import llama_cpp  # noqa: F401
        _LLAMACPP_AVAILABLE = True
    except ImportError:
        logger.warning(
            "llama-cpp-python is not installed. Install with: pip install hexonit-llm[llamacpp]"
        )
        _LLAMACPP_AVAILABLE = False
    except Exception as exc:
        logger.warning("llama-cpp-python import failed: %s", exc)
        _LLAMACPP_AVAILABLE = False

    return _LLAMACPP_AVAILABLE


# ── GGUF model resolution ───────────────────────────────────

def _resolve_gguf_path(
    model_id: str,
) -> Path:
    """
    Try to find a local GGUF file for *model_id*.

    Resolution order:
    1. If *model_id* is an existing file path, return it as-is.
    2. If it looks like a Hugging Face repo ID, look in the HF cache for
       any ``*.gguf`` file.
    3. Fall back to the model ID string (the engine will try to download).

    .. note::
        For full automation we rely on the user having converted their model
        to GGUF, or on ``llama-cpp-python``'s built-in support for HF repos
        (which auto-downloads GGUF files from TheBloke etc.).
    """
    p = Path(model_id)
    if p.exists():
        return p.resolve()

    # Check HF cache for GGUF files
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    model_slug = model_id.replace("/", "--")
    cache_dir = hf_home / ("models--" + model_slug)

    if cache_dir.exists():
        gguf_files = list(cache_dir.rglob("*.gguf"))
        if gguf_files:
            chosen = gguf_files[0]
            logger.info("Found cached GGUF: %s", chosen)
            return chosen

    # Not found locally – return the original ID; llama-cpp-python
    # will attempt to download from Hugging Face.
    logger.info("No local GGUF found for '%s'; will attempt download.", model_id)
    return p


# ── Public engine class ─────────────────────────────────────

class LlamaCppEngine:
    """
    Thin wrapper around ``llama_cpp.Llama`` with speculative decoding and
    maximum-throughput settings.

    Usage
    -----
    .. code-block:: python

        engine = LlamaCppEngine("meta-llama/Meta-Llama-3-70B-Instruct")
        outputs = engine.generate(["Your prompt here"])
    """

    def __init__(
        self,
        model: str,
        draft_model: Optional[str] = None,
        gguf_model_path: Optional[str] = None,
        cloud_draft_client: Optional[CloudDraftClient] = None,
    ) -> None:
        if not _check_llamacpp_available():
            raise LlamaCppUnavailableError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install hexonit-llm[llamacpp]"
            )

        self.model_name = model
        self._config = LlamaCppConfig(draft_model=draft_model)
        self._cloud_draft_client = cloud_draft_client

        # Resolve GGUF path
        model_path_str = gguf_model_path or model
        self._model_path = _resolve_gguf_path(model_path_str)
        self._draft_model_path: Optional[Path] = None

        self._llm: Any = None  # lazy init

    # ── Lazy initialisation ──────────────────────────────────

    def _init_llm(self) -> None:
        """Create the ``llama_cpp.Llama`` instance on first ``generate``."""
        if self._llm is not None:
            return

        import llama_cpp  # lazy import

        kwargs: dict[str, Any] = {
            "model_path": str(self._model_path),
            "n_gpu_layers": self._config.n_gpu_layers,
            "n_ctx": self._config.n_ctx,
            "n_batch": self._config.n_batch,
            "n_ubatch": self._config.n_ubatch,
            "n_threads": self._config.n_threads,
            "use_mmap": self._config.use_mmap,
            "use_mlock": self._config.use_mlock,
            "verbose": self._config.verbose,
        }

        # Flash attention (Metal on macOS uses it automatically)
        if self._config.flash_attn:
            kwargs["flash_attn"] = True

        # Speculative decoding – llama.cpp accepts a draft model path
        if self._config.draft_model is not None:
            draft_path = _resolve_gguf_path(self._config.draft_model)
            kwargs["draft_model"] = str(draft_path)

        # Merge any extras
        kwargs.update(self._config.extra_kwargs)

        logger.info(
            "Initialising llama.cpp engine (model=%s, draft=%s)",
            self.model_name,
            self._config.draft_model or "none",
        )
        self._llm = llama_cpp.Llama(**kwargs)

    # ── Public generate ──────────────────────────────────────

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
            Overrides for generation parameters (e.g. ``temperature=0.1``,
            ``max_tokens=4096``).

        Returns
        -------
        list[str]
            Generated text for each prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        self._init_llm()

        results: list[str] = []
        for prompt in prompts:
            output = self._llm.create_completion(
                prompt,
                temperature=sampling_kwargs.get("temperature", 0.7),
                top_p=sampling_kwargs.get("top_p", 0.9),
                max_tokens=sampling_kwargs.get("max_tokens", 2048),
                stop=sampling_kwargs.get("stop", None),
            )
            text = output.get("choices", [{}])[0].get("text", "")
            results.append(text)

        return results

    # ── Chat interface ───────────────────────────────────────

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        **sampling_kwargs: Any,
    ) -> str:
        """
        OpenAI-compatible chat completion.

        Parameters
        ----------
        messages : list[dict]
            Standard chat messages, e.g. ``[{"role": "user", "content": "..."}]``.
        **sampling_kwargs
            Overrides for generation parameters.

        Returns
        -------
        str
            Assistant reply content.
        """
        self._init_llm()

        output = self._llm.create_chat_completion(
            messages=messages,
            temperature=sampling_kwargs.get("temperature", 0.7),
            max_tokens=sampling_kwargs.get("max_tokens", 2048),
        )
        return output.get("choices", [{}])[0].get("message", {}).get("content", "")