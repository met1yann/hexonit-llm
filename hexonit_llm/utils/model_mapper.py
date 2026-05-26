"""
Model mapping utility – downloads / validates draft models from Hugging Face Hub.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfApi

from hexonit_llm.config.model_mappings import resolve_draft_model, TARGET_TO_DRAFT

logger = logging.getLogger("hexonit_llm")

# Default cache location (overridable via env var HF_HOME or XDG_CACHE_HOME)
DEFAULT_CACHE_DIR = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
)


def get_draft_model_name(target_model: str) -> str | None:
    """
    Return the Hugging Face repo ID of the optimal draft model for *target_model*.

    Mapping is purely a suggestion — no model is ever blocked.
    If no mapping exists, returns ``None`` silently.
    """
    draft = resolve_draft_model(target_model)
    if draft is not None:
        return draft

    # If the target itself is already a known draft model, return it as-is
    if target_model in TARGET_TO_DRAFT.values():
        return target_model

    # No mapping found — not a problem. Returns None silently.
    return None


def ensure_draft_model(
    draft_model_id: str,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Download (or verify cached) draft model from Hugging Face Hub.

    Parameters
    ----------
    draft_model_id : str
        Hugging Face repo ID, e.g. ``"meta-llama/Llama-3.2-3B-Instruct"``.
    cache_dir : Path, optional
        Override the default cache directory.
    force_download : bool
        If ``True``, re-download even if the model is already cached.

    Returns
    -------
    Path
        Local path to the downloaded model directory.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR

    # Check if already cached (quick path)
    api = HfApi()
    model_info = api.model_info(draft_model_id, token=None)
    cache_path = cache / ("models--" + model_info.id.replace("/", "--"))

    if not force_download and cache_path.exists():
        logger.info("Draft model already cached at %s", cache_path)
        return cache_path

    logger.info("Downloading draft model %s ...", draft_model_id)
    local_path = snapshot_download(
        repo_id=draft_model_id,
        cache_dir=str(cache),
        local_files_only=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info("Draft model downloaded to %s", local_path)
    return Path(local_path)


def ensure_model_local(
    model_id: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Download a model from Hugging Face Hub if it is not already cached locally.

    Returns the local path to the cached model directory.
    """
    cache = cache_dir or DEFAULT_CACHE_DIR
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache),
        local_files_only=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(local_path)