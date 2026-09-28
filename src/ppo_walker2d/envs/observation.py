"""Observation transformations."""

from __future__ import annotations

import numpy as np


def flatten_observation(observation: np.ndarray) -> np.ndarray:
    """Return a contiguous one-dimensional float32 observation."""
    return np.asarray(observation, dtype=np.float32).reshape(-1).copy()
