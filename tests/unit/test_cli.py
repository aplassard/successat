"""Unit tests for the successat command-line interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import pytest

from successat.benchmarks.base import Benchmark, BenchmarkExample, BenchmarkResult


@dataclass
class _DummyInnerClient:
    """Minimal inner client exposing a close method for cleanup."""

    closed: bool = False

    def close(self) -> None:
        self.closed = True


class _DummyClient:
    """Client double capturing constructor arguments."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        app_name: str | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or "dummy-model"
        self.app_name = app_name or "dummy-app"
        self.client_kwargs = dict(client_kwargs or {})
        self._inner = _DummyInnerClient()

    @property
    def client(self) -> _DummyInnerClient:
        return self._inner


def _build_result() -> BenchmarkResult:
    return BenchmarkResult(
        benchmark="gsm8k",
        model="dummy-model",
        split="test",
        example_id="0",
        prompt="Prompt text",
        response="raw",
        response_text="model response",
        correct=True,
        metadata={"expected": "model response"},
    )


def test_cli_runs_benchmark(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import successat.cli as cli

    recorded: Dict[str, Any] = {}

    def fake_run_benchmark(
        client: _DummyClient,
        benchmark_name: str,
        *,
        identifier: int | str | None,
        split: str | None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        recorded["client"] = client
        recorded["benchmark"] = benchmark_name
        recorded["identifier"] = identifier
        recorded["split"] = split
        recorded["kwargs"] = kwargs
        return _build_result()

    monkeypatch.setattr(cli, "CLIENT_FACTORIES", {"dummy": _DummyClient})
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(
        [
            "--benchmark",
            "gsm8k",
            "--client",
            "dummy",
            "--api-key",
            "test-key",
            "--model",
            "custom-model",
            "--split",
            "validation",
            "--identifier",
            "3",
            "--param",
            "temperature=0.2",
            "--param",
            "echo=true",
            "--client-option",
            "timeout=30",
        ]
    )

    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Benchmark: gsm8k" in out
    assert "Response:\nmodel response" in out

    client = recorded["client"]
    assert isinstance(client, _DummyClient)
    assert client.api_key == "test-key"
    assert client.model == "custom-model"
    assert client.client_kwargs == {"timeout": 30}

    assert recorded["benchmark"] == "gsm8k"
    assert recorded["identifier"] == 3
    assert recorded["split"] == "validation"
    assert recorded["kwargs"] == {"temperature": 0.2, "echo": True}
    assert client.client.closed is True


def test_cli_lists_benchmarks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import successat.cli as cli

    monkeypatch.setattr(cli, "benchmark_registry", type("Registry", (), {"names": lambda self: ["b", "a"]})())

    exit_code = cli.main(["--list-benchmarks"])
    assert exit_code == 0

    out = capsys.readouterr().out.splitlines()
    assert out == ["b", "a"]


def test_cli_shows_benchmark_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import successat.cli as cli

    class _DetailBenchmark(Benchmark):
        name = "demo"
        default_split = "b"

        def available_splits(self) -> list[str]:
            return ["a", "b"]

        def examples_for_split(self, split: str):
            if split == "a":
                return [
                    BenchmarkExample(id="a1", prompt="", target=""),
                    BenchmarkExample(id="a2", prompt="", target=""),
                ]
            if split == "b":
                return [BenchmarkExample(id="b1", prompt="", target="")]
            raise ValueError(split)

    class _Registry:
        def names(self) -> list[str]:
            return ["demo"]

        def get(self, name: str):
            if name == "demo":
                return _DetailBenchmark
            raise KeyError(name)

    monkeypatch.setattr(cli, "benchmark_registry", _Registry())

    exit_code = cli.main(["--benchmark-details", "demo"])
    assert exit_code == 0

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Benchmark: demo"
    assert out[1] == "Splits:"
    assert out[2] == "  - a: 2 examples"
    assert out[3] == "  - b (default): 1 example"


def test_cli_reports_invalid_parameter(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import successat.cli as cli

    monkeypatch.setattr(cli, "CLIENT_FACTORIES", {"dummy": _DummyClient})

    exit_code = cli.main(
        ["--benchmark", "gsm8k", "--client", "dummy", "--api-key", "key", "--param", "invalid"]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Invalid parameter" in err
