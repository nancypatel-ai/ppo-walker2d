"""Observation tests."""

import numpy as np

from ppo_walker2d.envs.observation import flatten_observation


def test_flatten_observation_returns_float32_copy() -> None:
    source = np.array([[1, 2]], dtype=np.int64)
    result = flatten_observation(source)
    assert result.dtype == np.float32
    assert result.shape == (2,)
    assert not np.shares_memory(source, result)
