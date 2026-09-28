"""Local Gradio demonstration entry point."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    try:
        import gradio as gr
    except ImportError as error:
        raise SystemExit(
            "Install the demo extra with: pip install -e '.[demo]'"
        ) from error

    def show_checkpoint() -> str:
        return f"Checkpoint: {args.checkpoint}"

    gr.Interface(
        fn=show_checkpoint, inputs=None, outputs="text", title="ppo-walker2d"
    ).launch(
        analytics_enabled=False
    )


if __name__ == "__main__":
    main()
