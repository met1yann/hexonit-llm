"""
Speculative Decoding Model Mappings.

Maps large target models to their optimal draft (assistant) models.
All draft models share the same tokenizer as their target counterpart.
"""

# ──────────────────────────────────────────────────────────────
# Official target → draft model mappings for speculative decoding
# ──────────────────────────────────────────────────────────────
TARGET_TO_DRAFT: dict[str, str] = {
    # ── Meta LLaMA Family ────────────────────────────────────
    "meta-llama/Meta-Llama-3-70B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-2-70B-chat-hf": "meta-llama/Llama-2-7B-chat-hf",
    "meta-llama/Llama-2-13B-chat-hf": "meta-llama/Llama-2-7B-chat-hf",

    # ── Qwen Family ──────────────────────────────────────────
    "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2-72B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-57B-A14B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/QwQ-32B-Preview": "Qwen/Qwen2.5-1.5B-Instruct",

    # ── Mistral / Mixtral Family ─────────────────────────────
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Large-Instruct-2407": "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",

    # ── Google Gemma Family ──────────────────────────────────
    "google/gemma-2-27b-it": "google/gemma-2-2b-it",
    "google/gemma-2-9b-it": "google/gemma-2-2b-it",
    "google/gemma-2-2b-it": "google/gemma-2-2b-it",

    # ── CodeLLaMA Family ─────────────────────────────────────
    "codellama/CodeLlama-34b-Instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-7b-Instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",

    # ── DeepSeek Family ──────────────────────────────────────
    "deepseek-ai/DeepSeek-V2.5": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-ai/deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-7b-chat",

    # ── Microsoft Phi Family ─────────────────────────────────
    "microsoft/Phi-3-medium-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",

    # ── IBM Granite Family ───────────────────────────────────
    "ibm-granite/granite-34b-code-instruct": "ibm-granite/granite-8b-code-instruct",
    "ibm-granite/granite-20b-code-instruct": "ibm-granite/granite-8b-code-instruct",
    "ibm-granite/granite-8b-code-instruct": "ibm-granite/granite-8b-code-instruct",
}

# ── Model name normalisation helpers ────────────────────────
def normalize_model_name(name: str) -> str:
    """Strip optional prefixes/suffixes so we can match user input against our map."""
    cleaned = name.strip().removeprefix("hf:").removeprefix("https://huggingface.co/")
    if cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned


def resolve_draft_model(target_model: str) -> str | None:
    """
    Return the best draft model for *target_model*, or ``None`` if no mapping exists.

    Example:
        >>> resolve_draft_model("meta-llama/Meta-Llama-3-70B-Instruct")
        'meta-llama/Llama-3.2-3B-Instruct'

        >>> resolve_draft_model("unknown/model") is None
        True
    """
    key = normalize_model_name(target_model)
    return TARGET_TO_DRAFT.get(key)