"""Small local logger with no network or telemetry behavior."""

from __future__ import annotations

import json
import os
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


class WandbLogger:
    """Opt-in W&B adapter that is inert when no project is supplied."""

    def __init__(self, project: str | None, run_name: str) -> None:
        self.run = None
        if project is not None:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError(
                    "Install the tracking extra to enable W&B logging."
                ) from error
            entity = os.getenv("WANDB_ENTITY")
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                reinit="create_new",
            )

    def write(self, metrics: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(metrics)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
