"""Environment construction and optional domain randomization."""

from __future__ import annotations

from typing import Any

from ppo_walker2d.envs.reward import reward_v1, reward_v2


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
    environment_instance = RewardWrapper(environment_instance, reward_name)
    return environment_instance


def make_vector_env(
    environment: str,
    num_envs: int,
    seed: int,
    reward_name: str = "reward_v1",
    domain_randomization: bool = False,
) -> Any:
    """Create a synchronous vector environment with independent seeds."""
    import gymnasium as gym

    def factory(index: int) -> Any:
        return lambda: make_env(
            environment,
            seed + index,
            reward_name,
            domain_randomization,
        )

    return gym.vector.SyncVectorEnv([factory(index) for index in range(num_envs)])


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


class RewardWrapper:
    """Replace the environment reward with one of the project reward variants."""

    def __init__(self, environment: Any, reward_name: str) -> None:
        if reward_name not in {"reward_v1", "reward_v2"}:
            raise ValueError(f"Unknown reward: {reward_name}")
        self.environment = environment
        self.reward_name = reward_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, _, terminated, truncated, info = self.environment.step(action)
        model = self.environment.unwrapped
        forward_velocity = float(model.data.qvel[0])
        control_cost = 0.001 * float((action**2).sum())
        is_healthy = bool(model.is_healthy)
        reward_function = reward_v1 if self.reward_name == "reward_v1" else reward_v2
        reward = reward_function(forward_velocity, control_cost, is_healthy)
        info = dict(info)
        info.update({"forward_velocity": forward_velocity, "healthy": is_healthy})
        return observation, reward, terminated, truncated, info
