"""Environment construction and optional domain randomization."""

from __future__ import annotations

from typing import Any


def make_env(
    environment: str,
    seed: int,
    reward_name: str = "reward_v1",
    domain_randomization: bool = False,
) -> Any:
    """Create one seeded Gymnasium environment.

    The import stays local so reward and model unit tests do not require MuJoCo.
    """
    import gymnasium as gym

    environment_instance: Any = gym.make(environment)
    environment_instance.reset(seed=seed)
    environment_instance.action_space.seed(seed)
    if domain_randomization:
        environment_instance = DomainRandomizationWrapper(environment_instance, seed)
    environment_instance.reward_name = reward_name
    return environment_instance


class DomainRandomizationWrapper:
    """Apply deterministic mass and friction perturbations at each reset."""

    def __init__(self, environment: Any, seed: int) -> None:
        self.environment = environment
        self.rng = __import__("numpy").random.default_rng(seed)
        self.base_mass = environment.unwrapped.model.body_mass.copy()
        self.base_friction = environment.unwrapped.model.geom_friction.copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        model = self.environment.unwrapped.model
        model.body_mass[:] = self.base_mass
        model.geom_friction[:] = self.base_friction
        model.body_mass[:] *= self.rng.uniform(0.8, 1.2, size=model.body_mass.shape)
        model.geom_friction[:] *= self.rng.uniform(
            0.7, 1.3, size=model.geom_friction.shape
        )
        return self.environment.reset(**kwargs)
