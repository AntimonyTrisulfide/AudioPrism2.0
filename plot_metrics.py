from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AudioPrism training curves")
    parser.add_argument("metrics", type=Path, help="Path to logs/metrics.jsonl")
    parser.add_argument("--output", type=Path, default=Path("training_curves.png"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [record["epoch"] + 1 for record in records]
    train_loss = [record["train"]["loss"] for record in records]
    val_loss = [record["validation"]["loss"] if record["validation"] else None for record in records]
    val_sisdr = [record["validation"]["si_sdr"] if record["validation"] else None for record in records]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Objective", title="Separation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, val_sisdr, color="#b33a3a")
    axes[1].set(xlabel="Epoch", ylabel="SI-SDR (dB)", title="Validation SI-SDR")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
