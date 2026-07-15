from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audioprism.data import PreprocessedComplexDataset
from audioprism.engine import validate
from audioprism.factory import model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AudioPrism 2.0 checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, source_names, checkpoint = model_from_checkpoint(args.checkpoint, device)
    dataset = PreprocessedComplexDataset(
        args.data_dir, config.data.cache_tracks, 0.0, config.data.allow_mixture_phase_targets
    )
    if dataset.source_names != source_names:
        raise ValueError("Checkpoint and evaluation dataset use different source ordering")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    metrics = validate(model, loader, device, config, len(source_names))
    per_source_si_sdr = metrics.pop("per_source_si_sdr")
    per_source_snr = metrics.pop("per_source_snr")
    per_source_active = metrics.pop("per_source_active_chunks")
    metrics["per_source"] = {
        name: {
            "si_sdr": per_source_si_sdr[index],
            "snr": per_source_snr[index],
            "active_chunks": per_source_active[index],
        }
        for index, name in enumerate(source_names)
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "data_dir": str(args.data_dir.resolve()),
        "chunks": len(dataset),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
