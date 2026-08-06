"""High-resolution timing helpers."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType


@dataclass
class Timer(AbstractContextManager["Timer"]):
    elapsed_ms: float = 0.0
    _start_ns: int = 0

    def __enter__(self) -> Timer:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000
