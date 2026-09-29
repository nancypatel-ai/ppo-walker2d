"""Local Gradio demonstration entry point."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import torch

from ppo_walker2d.config import load_config
from ppo_walker2d.models.actor_critic import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/reward_v1__seed42.pt")
    parser.add_argument("--config", default="configs/reward_v1.yaml")
    args = parser.parse_args()
    try:
        import gradio as gr
    except ImportError as error:
        raise SystemExit(
            "Install the demo extra with: pip install -e '.[demo]'"
        ) from error

    config = load_config(args.config)
    environment = __import__("gymnasium").make(
        config.environment, render_mode="rgb_array"
    )
    assert environment.observation_space.shape is not None
    assert environment.action_space.shape is not None
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = ActorCritic(
        environment.observation_space.shape[0],
        environment.action_space.shape[0],
        config.training.hidden_sizes,
    )
    policy.load_state_dict(checkpoint["model"])

    def show_checkpoint() -> str:
        renderer = mujoco.Renderer(environment.unwrapped.model, height=240, width=420)
        frames = []
        observation, _ = environment.reset(seed=config.seed)
        for _ in range(200):
            renderer.update_scene(environment.unwrapped.data)
            frames.append(renderer.render().copy())
            with torch.no_grad():
                action = policy.deterministic_action(
                    torch.as_tensor(observation, dtype=torch.float32)
                )
            observation, _, terminated, truncated, _ = environment.step(action.numpy())
            if terminated or truncated:
                break
        output = Path("outputs") / "demo.gif"
        output.parent.mkdir(exist_ok=True)
        imageio.mimsave(output, frames, duration=0.04, loop=0)
        renderer.close()
        return str(output)

    gr.Interface(
        fn=show_checkpoint, inputs=None, outputs="video", title="ppo-walker2d"
    ).launch(
        analytics_enabled=False
    )


if __name__ == "__main__":
    main()
