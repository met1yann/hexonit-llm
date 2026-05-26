"""
Orchestrator – the intelligent router that selects & initialises the optimal
inference engine based on live hardware inspection.

Triple-mode routing:

**Mode 1 – Cloud-Assisted Draft (Hybrid)**
    User provides ``draft_base_url`` + ``api_key``.
    → Target model loads locally; draft tokens come from ANY OpenAI-compatible
      API endpoint (Groq, OpenRouter, Together, SambaNova, local vLLM, etc.).
    → If network fails → graceful fallback to single-model local inference.

**Mode 2 – Explicit Local Pair (max speed)**
    User provides ``draft_model`` explicitly, leaves ``draft_base_url=None``.
    → BOTH models load locally for speculative decoding.
    → Bypasses all mapping dictionaries.

**Mode 3 – Auto Local (default, zero config)**
    User provides only ``model``.
    → Auto-resolve best draft from ``model_mappings.py``.
    → If unknown → single-model local inference.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hexonit_llm.utils.hardware_detector import (
    detect_hardware,
    select_engine,
    HardwareProfile,
)
from hexonit_llm.utils.model_mapper import get_draft_model_name

logger = logging.getLogger("hexonit_llm.orchestrator")


class Orchestrator:
    """
    Manages the lifecycle of the chosen inference engine.

    Parameters
    ----------
    model : str
        Hugging Face repo ID or local path.
    optimization_level : str
        One of ``"max_speed"``, ``"balanced"``, ``"memory_saver"``.
    explicit_draft_model : str, optional
        Explicit local draft model (Mode 2). Overrides auto-mapping.
    draft_base_url : str, optional
        Base URL of any OpenAI-compatible chat completions API (Mode 1).
        Requires ``api_key``.
    api_key : str, optional
        API key for the remote endpoint.
    **engine_kwargs
        Additional keyword arguments for the engine constructor.
    """

    def __init__(
        self,
        model: str,
        optimization_level: str = "max_speed",
        explicit_draft_model: Optional[str] = None,
        draft_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **engine_kwargs: Any,
    ) -> None:
        self.model = model
        self.optimization_level = optimization_level
        self._engine_kwargs = engine_kwargs
        self.explicit_draft_model = explicit_draft_model
        self.draft_base_url = draft_base_url
        self.api_key = api_key

        # Operating mode
        self.mode: str = "unknown"

        # ── 1. Detect hardware ───────────────────────────────
        logger.info("Detecting hardware …")
        self.hw: HardwareProfile = detect_hardware()
        logger.info(
            "Hardware: OS=%s | VRAM=%.1f GB | RAM=%.1f GB | CUDA=%s",
            self.hw.os_type,
            self.hw.total_vram_gb,
            self.hw.total_ram_gb,
            self.hw.cuda_available,
        )

        # ── 2. Select engine ─────────────────────────────────
        self.engine_name: str = select_engine()
        logger.info("Selected engine: %s", self.engine_name)

        # ── 3. Resolve draft model ────────────────────────────
        self.draft_model: Optional[str] = None
        self.cloud_draft_client: Any = None

        self._resolve_draft()

        # ── 4. Lazy engine ────────────────────────────────────
        self._engine: Any = None

    def _resolve_draft(self) -> None:
        """Determine the draft source based on user input."""

        # ── MODE 1: Cloud-Assisted Draft (Universal OpenAI API) ──
        if self.draft_base_url is not None and self.api_key is not None:
            self.mode = "hybrid_cloud"
            logger.info(
                "Cloud draft mode: base_url=%s, api_key=***%s",
                self.draft_base_url,
                self.api_key[-4:] if len(self.api_key) >= 4 else "",
            )

            from hexonit_llm.engines.cloud_draft_provider import CloudDraftClient

            # Model name: use draft_model if provided, else user's target model
            cloud_model = self.explicit_draft_model or self.model

            try:
                self.cloud_draft_client = CloudDraftClient(
                    base_url=self.draft_base_url,
                    api_key=self.api_key,
                    model=cloud_model,
                )
                self.draft_model = cloud_model
                logger.info(
                    "[Hexonithy Studios] Cloud draft ready — "
                    "base_url=%s | model=%s",
                    self.draft_base_url,
                    cloud_model,
                )
            except Exception as exc:
                logger.warning(
                    "[Hexonithy Studios] Cloud draft init failed: %s. "
                    "Degrading to single-model local inference.",
                    exc,
                )
                self.cloud_draft_client = None
                self.draft_model = None
                self.mode = "local_single"
            return

        # ── MODE 2: Explicit Local Pair ───────────────────────
        if self.explicit_draft_model is not None:
            self.mode = "local_explicit_pair"
            self.draft_model = self.explicit_draft_model
            logger.info(
                "Explicit local draft pair: target=%s, draft=%s",
                self.model,
                self.draft_model,
            )
            return

        # ── MODE 3: Auto Local ────────────────────────────────
        # Mapping is a suggestion, never a blocker.
        # If no mapping found, runs single-model silently.
        auto_draft = get_draft_model_name(self.model)
        if auto_draft is not None:
            self.mode = "local_auto"
            self.draft_model = auto_draft
            logger.info(
                "Auto-mapped local draft: %s → %s",
                self.model,
                self.draft_model,
            )
        else:
            self.mode = "local_single"
            self.draft_model = None

    # ── Engine factory (lazy) ────────────────────────────────

    def _get_engine(self) -> Any:
        """Build the engine on first call; cached thereafter."""
        if self._engine is not None:
            return self._engine

        if self.engine_name == "vllm":
            self._engine = self._build_vllm()
        else:
            self._engine = self._build_llamacpp()

        return self._engine

    def _build_vllm(self) -> Any:
        """Create a ``VllmEngine`` instance (with optional cloud draft)."""
        from hexonit_llm.engines.vllm_engine import VllmEngine, VllmUnavailableError

        try:
            return VllmEngine(
                model=self.model,
                draft_model=self.draft_model,
                cloud_draft_client=self.cloud_draft_client,
                **self._engine_kwargs,
            )
        except VllmUnavailableError:
            logger.warning(
                "[Hexonithy Studios] vLLM unavailable despite hardware check. "
                "Falling back to llama.cpp engine."
            )
            self.engine_name = "llamacpp"
            return self._build_llamacpp()

    def _build_llamacpp(self) -> Any:
        """Create a ``LlamaCppEngine`` instance (with optional cloud draft)."""
        from hexonit_llm.engines.llamacpp_engine import (
            LlamaCppEngine,
            LlamaCppUnavailableError,
        )

        try:
            return LlamaCppEngine(
                model=self.model,
                draft_model=self.draft_model,
                cloud_draft_client=self.cloud_draft_client,
                **self._engine_kwargs,
            )
        except LlamaCppUnavailableError as exc:
            raise RuntimeError(
                "[Hexonithy Studios] Neither vLLM nor llama-cpp-python is available.\n"
                "  Install at least: pip install hexonit-llm[vllm]  (Linux)\n"
                "  or:                pip install hexonit-llm[llamacpp]  (all platforms)\n"
            ) from exc

    # ── Public API ───────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        **sampling_kwargs: Any,
    ) -> str:
        """
        Generate a completion for a single prompt.

        In cloud-assisted mode, the target model runs locally while draft
        tokens come from the remote OpenAI-compatible endpoint.
        """
        engine = self._get_engine()
        results = engine.generate([prompt], **sampling_kwargs)
        return results[0]

    def generate_batch(
        self,
        prompts: list[str],
        **sampling_kwargs: Any,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.
        """
        engine = self._get_engine()
        return engine.generate(prompts, **sampling_kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        **sampling_kwargs: Any,
    ) -> str:
        """
        OpenAI-compatible chat interface.
        """
        engine = self._get_engine()
        if hasattr(engine, "create_chat_completion"):
            return engine.create_chat_completion(messages, **sampling_kwargs)

        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, **sampling_kwargs)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Simple chat-to-text conversion for engines without native chat."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        lines.append("Assistant: ")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Orchestrator(model={self.model!r}, "
            f"engine={self.engine_name!r}, "
            f"mode={self.mode!r}, "
            f"draft={self.draft_model!r})"
        )