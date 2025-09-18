"""Tests for the dataset loading helpers."""

from __future__ import annotations

import pytest

import successat.datasets as datasets_module


@pytest.fixture(autouse=True)
def reset_cache_env(monkeypatch):
    """Ensure cache env var is not leaked between tests."""

    monkeypatch.delenv("SUCCESSAT_CACHE_DIR", raising=False)


def test_load_dataset_uses_successat_cache_dir(tmp_path, monkeypatch):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"name": "dummy"}

    monkeypatch.setenv("SUCCESSAT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(datasets_module, "_HF_LOAD_DATASET", fake_load_dataset)

    result = datasets_module.load_dataset("dummy")

    assert result == {"name": "dummy"}
    expected_dir = tmp_path / "datasets"
    assert captured["kwargs"]["cache_dir"] == str(expected_dir)
    assert expected_dir.exists()
    assert captured["kwargs"]["download_mode"] == "reuse_cache_if_exists"


def test_load_dataset_respects_explicit_cache_dir(tmp_path, monkeypatch):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"name": "explicit"}

    monkeypatch.setattr(datasets_module, "_HF_LOAD_DATASET", fake_load_dataset)

    custom_cache = tmp_path / "custom"
    result = datasets_module.load_dataset("dummy", cache_dir=custom_cache)

    assert result == {"name": "explicit"}
    assert captured["kwargs"]["cache_dir"] == str(custom_cache)
    assert custom_cache.exists()
