"""Abstract base class for all inference engines."""
from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Base interface that all inference engines must implement."""

    def __init__(self, model_name: str, draft_model: str | None = None, **kwargs: Any) -> None:
        self.model_name = model_name
        self.draft_model = draft_model
        self._config_extra: dict[str, Any] = kwargs

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def generate_batch(self, prompts: list[str], max_tokens: int = 512, **kwargs: Any) -> list[str]:
        """Generate text for multiple prompts."""
        ...

    @abstractmethod
    def chat(self, messages: list[dict], max_tokens: int = 512, **kwargs: Any) -> str:
        """Chat interface with message history."""
        ...

    @abstractmethod
    def benchmark(self, prompt: str = "Tell me about artificial intelligence.", runs: int = 5) -> dict:
        """Run benchmark and return tokens/sec stats."""
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return engine identifier string."""
        ...