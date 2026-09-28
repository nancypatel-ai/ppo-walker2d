"""Generate the project learning-curve artifact from local JSONL logs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    logs = sorted(Path("outputs").glob("*.jsonl"))
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=160)
    for path in logs:
        steps: list[int] = []
        values: list[float] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "policy_loss" in record:
                steps.append(int(record["step"]))
                values.append(float(record["policy_loss"]))
        if steps:
            axis.plot(steps, values, label=path.stem, linewidth=1.5)
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Policy loss")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if logs:
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    figure.savefig("assets/learning_curves.png", transparent=False)
    plt.close(figure)


if __name__ == "__main__":
    main()
