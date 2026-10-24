"""Bounded resource controls, LRU caching, and reproducible benchmarks."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceBudget:
    max_context_tokens: int = 8_000
    max_output_bytes: int = 5_000_000
    max_memory_bytes: int = 2_000_000_000
    max_wall_seconds: float = 600.0

    @staticmethod
    def estimate_tokens(value: Any) -> int:
        return max(1, len(json.dumps(value, ensure_ascii=False, default=str)) // 4)

    def validate(
        self,
        *,
        context: Any = None,
        output_bytes: int = 0,
        estimated_memory_bytes: int = 0,
        elapsed_seconds: float = 0,
    ) -> None:
        if context is not None and self.estimate_tokens(context) > self.max_context_tokens:
            raise BudgetExceeded("context_token_budget_exceeded")
        if output_bytes > self.max_output_bytes:
            raise BudgetExceeded("output_size_budget_exceeded")
        if estimated_memory_bytes > self.max_memory_bytes:
            raise BudgetExceeded("memory_budget_exceeded")
        if elapsed_seconds > self.max_wall_seconds:
            raise BudgetExceeded("wall_time_budget_exceeded")


class LRUCache:
    def __init__(self, *, max_items: int = 128, max_bytes: int = 64_000_000):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return item[0]

    def put(self, key: str, value: Any, *, size_bytes: int | None = None) -> bool:
        size = size_bytes if size_bytes is not None else len(repr(value).encode())
        if size > self.max_bytes:
            return False
        previous = self._items.pop(key, None)
        if previous:
            self._bytes -= previous[1]
        self._items[key] = (value, size)
        self._bytes += size
        while len(self._items) > self.max_items or self._bytes > self.max_bytes:
            _, (_, removed) = self._items.popitem(last=False)
            self._bytes -= removed
        return True

    @property
    def size(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class BenchmarkResult:
    tier: str
    elapsed_seconds: float
    passed: bool


def run_benchmark(tier: str, operation: Callable[[], Any], max_seconds: float) -> BenchmarkResult:
    if tier not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("Unknown benchmark tier")
    started = time.perf_counter()
    operation()
    elapsed = time.perf_counter() - started
    return BenchmarkResult(tier=tier, elapsed_seconds=elapsed, passed=elapsed <= max_seconds)


def backed_anndata_summary(path: str | Path) -> dict[str, Any]:
    """Read only structural metadata without materializing the expression matrix."""
    import anndata as ad

    data = ad.read_h5ad(Path(path), backed="r")
    try:
        return {
            "n_obs": int(data.n_obs),
            "n_vars": int(data.n_vars),
            "obs_columns": [str(value) for value in data.obs.columns],
            "var_columns": [str(value) for value in data.var.columns],
            "backed": bool(data.isbacked),
        }
    finally:
        data.file.close()
