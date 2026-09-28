"""Small local logger with no network or telemetry behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Write one JSON object per training iteration."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = Path(path).open("w", encoding="utf-8")

    def write(self, metrics: dict[str, Any]) -> None:
        self.handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()
