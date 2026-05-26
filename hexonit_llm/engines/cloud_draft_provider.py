"""
Universal OpenAI-Compatible Cloud Draft Provider.

This module provides a generic HTTP client that works with ANY OpenAI-compatible
API endpoint for draft token generation in speculative decoding.

Supported endpoints include (but are not limited to):
- Groq:       https://api.groq.com/openai/v1
- OpenRouter: https://openrouter.ai/api/v1
- Together:   https://api.together.xyz/v1
- SambaNova:  https://api.sambanova.ai/v1
- Local:      http://localhost:8000/v1  (self-hosted vLLM/Ollama)
- Custom:     any OpenAI-compatible chat completions endpoint

The heavy primary model always runs locally. Draft tokens are fetched
from the remote endpoint at each speculative decoding step.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("hexonit_llm.engines.cloud_draft")


class CloudDraftClient:
    """
    Generic OpenAI-compatible HTTP client for draft token generation.

    Works with ANY provider that exposes an OpenAI-compatible
    ``/chat/completions`` endpoint.

    Parameters
    ----------
    base_url : str
        Base URL of the OpenAI-compatible API.
        Examples:
        - ``"https://api.groq.com/openai/v1"``
        - ``"https://openrouter.ai/api/v1"``
        - ``"http://localhost:8000/v1"``
    api_key : str
        API key for the endpoint.
    model : str
        Model identifier on the remote endpoint, e.g. ``"llama-3.1-8b-instant"``
        or ``"gpt-3.5-turbo"``.
    timeout : int
        HTTP request timeout in seconds (default 15).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._session: Any = None

    def _get_session(self) -> Any:
        """Lazy-create the httpx client."""
        if self._session is None:
            try:
                import httpx
            except ImportError:
                raise RuntimeError(
                    "httpx is required for cloud draft provider. "
                    "Install with: pip install hexonit-llm[cloud]"
                )
            self._session = httpx.Client(timeout=self.timeout)
        return self._session

    def generate_draft(
        self,
        prompt: str,
        max_draft_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """
        Generate a draft completion from the remote endpoint.

        Parameters
        ----------
        prompt : str
            Input text to continue from.
        max_draft_tokens : int
            Maximum draft tokens to generate (default 128).
        temperature : float
            Sampling temperature (default 0.0 = greedy).

        Returns
        -------
        str or None
            Draft text on success, ``None`` on any failure (network, auth, etc.).
        """
        session = self._get_session()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Truncate prompt for API limits (most endpoints cap at ~32K chars)
        truncated_prompt = prompt[-32000:] if len(prompt) > 32000 else prompt

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": truncated_prompt}],
            "max_tokens": max_draft_tokens,
            "temperature": temperature,
            "stream": False,
        }

        chat_url = f"{self.base_url}/chat/completions"

        try:
            resp = session.post(chat_url, headers=headers, json=body)
            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                logger.debug(
                    "Cloud draft: %d tokens from %s",
                    len(content.split()),
                    self.base_url,
                )
                return content
            else:
                logger.warning(
                    "[Hexonithy Studios] Cloud draft API error (%d): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return None
        except Exception as exc:
            logger.warning(
                "[Hexonithy Studios] Cloud draft request failed: %s. "
                "Degrading to single-model local inference.",
                exc,
            )
            return None

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __repr__(self) -> str:
        return (
            f"CloudDraftClient(base_url={self.base_url!r}, "
            f"model={self.model!r})"
        )