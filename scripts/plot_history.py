"""Plot training and validation curves from history.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axes[0].plot(epochs, [entry["validation_loss"] for entry in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [entry["validation_dice"] for entry in history], label="validation Dice")
    axes[1].set_title("Foreground Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
