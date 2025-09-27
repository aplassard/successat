"""Concurrency-aware helpers for executing multiple benchmark runs."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, List, Mapping

from .base import BenchmarkRegistry, BenchmarkResult, SupportsChatCompletion


ResultCallback = Callable[["BenchmarkRunSpec", BenchmarkResult], Awaitable[None] | None]


@dataclass(slots=True)
class BenchmarkRunSpec:
    """Description of an individual benchmark invocation."""

    benchmark: str
    client: SupportsChatCompletion
    identifier: int | str | None = None
    split: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def resolved_parameters(self) -> dict[str, Any]:
        """Return a defensive copy of the parameters passed to ``run``."""

        if not self.parameters:
            return {}
        return dict(self.parameters)


class BenchmarkRunner:
    """Execute multiple benchmark runs with a configurable concurrency limit."""

    def __init__(
        self,
        *,
        registry: BenchmarkRegistry | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            msg = "max_concurrency must be at least 1 when provided"
            raise ValueError(msg)

        if registry is None:
            from . import benchmark_registry as _default_registry

            registry = _default_registry

        self._registry = registry
        self._semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )

    async def run_all_async(
        self,
        specs: Iterable[BenchmarkRunSpec],
        *,
        result_callback: ResultCallback | None = None,
    ) -> List[BenchmarkResult]:
        """Execute all specs concurrently while respecting the semaphore."""

        items = list(specs)
        tasks = [
            asyncio.create_task(self._run_single(spec, result_callback=result_callback))
            for spec in items
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    def run_all(
        self,
        specs: Iterable[BenchmarkRunSpec],
        *,
        result_callback: ResultCallback | None = None,
    ) -> List[BenchmarkResult]:
        """Synchronous wrapper around :meth:`run_all_async`."""

        return asyncio.run(self.run_all_async(specs, result_callback=result_callback))

    async def _run_single(
        self,
        spec: BenchmarkRunSpec,
        *,
        result_callback: ResultCallback | None,
    ) -> BenchmarkResult:
        if self._semaphore is None:
            result = await asyncio.to_thread(self._execute_spec, spec)
        else:
            async with self._semaphore:
                result = await asyncio.to_thread(self._execute_spec, spec)

        if result_callback is not None:
            await self._emit_callback(result_callback, spec, result)

        return result

    def _execute_spec(self, spec: BenchmarkRunSpec) -> BenchmarkResult:
        benchmark_cls = self._registry.get(spec.benchmark)
        runner = benchmark_cls(spec.client)
        params = spec.resolved_parameters()
        return runner.run(identifier=spec.identifier, split=spec.split, **params)

    async def _emit_callback(
        self,
        callback: ResultCallback,
        spec: BenchmarkRunSpec,
        result: BenchmarkResult,
    ) -> None:
        maybe = callback(spec, result)
        if inspect.isawaitable(maybe):
            await maybe


__all__ = ["BenchmarkRunSpec", "BenchmarkRunner", "ResultCallback"]

