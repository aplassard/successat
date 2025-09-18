"""Wrappers around dataset loading utilities with successat-specific defaults."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from datasets import load_dataset as _HF_LOAD_DATASET

from .cache import cache_subdir


def _prepare_cache_dir(cache_dir: str | PathLike[str] | None) -> Path:
    if cache_dir is None:
        return cache_subdir("datasets")
    path = Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(*args: Any, **kwargs: Any):
    """Load a dataset using HuggingFace with successat caching defaults.

    The cache directory defaults to ``~/.successat/cache/datasets`` but can be
    overridden either by passing ``cache_dir`` explicitly or by setting the
    ``SUCCESSAT_CACHE_DIR`` environment variable.
    """

    cache_path = _prepare_cache_dir(kwargs.pop("cache_dir", None))
    kwargs.setdefault("download_mode", "reuse_cache_if_exists")

    return _HF_LOAD_DATASET(*args, cache_dir=str(cache_path), **kwargs)
