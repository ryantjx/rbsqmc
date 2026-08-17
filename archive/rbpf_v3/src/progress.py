"""Flush-safe host progress logging used by local and Colab runners."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path


def progress(message: str, *, stream: str = "OUT") -> None:
    outer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inner = datetime.now().strftime("%H:%M:%S")
    line = f"[{outer}] {stream}: [{inner}] {message}"
    print(line, flush=True)
    log_path = os.environ.get("RBSQMC_PROGRESS_LOG")
    if log_path:
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
