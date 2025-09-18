"""Utilities for managing successat cache directories."""

from __future__ import annotations

import os
from pathlib import Path

_CACHE_ENV_VAR = "SUCCESSAT_CACHE_DIR"


def cache_root() -> Path:
    """Return the root directory for successat caches.

    The location can be overridden by setting the ``SUCCESSAT_CACHE_DIR``
    environment variable. When unset, the default is ``~/.successat/cache``.
    The directory is created if it does not already exist.
    """

    override = os.environ.get(_CACHE_ENV_VAR)
    if override:
        root = Path(override).expanduser()
    else:
        root = Path.home() / ".successat" / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_subdir(*parts: str) -> Path:
    """Return a subdirectory of the cache root, creating it if needed."""

    root = cache_root()
    if not parts:
        return root
    subdir = root.joinpath(*parts)
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir
