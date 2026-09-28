"""Evaluation metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvaluationMetrics:
    mean_return: float
    mean_distance: float
    mean_speed: float
    fall_rate: float
    energy_proxy: float
    episodes: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
        )
