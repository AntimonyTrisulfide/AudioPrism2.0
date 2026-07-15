from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audioprism.audio import istft
from audioprism.data import PreprocessedComplexDataset
from audioprism.metrics import si_sdr, snr


def multiply_real_mask(mask: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    return mixture[:, None] * mask[:, :, None]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure separation ceilings and baselines")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("oracle_ceiling.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    dataset = PreprocessedComplexDataset(args.data_dir, cache_tracks=2, gain_augmentation_db=0.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metadata = dataset.metadata
    totals: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(len(dataset.source_names)))
    counts = torch.zeros(len(dataset.source_names))
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        mixture = batch["mix"]
        targets = batch["targets"]
        activity = batch["activity"]
        target_mag = targets.square().sum(dim=2).sqrt()
        mix_mag = mixture.square().sum(dim=1).sqrt().clamp_min(1.0e-8)
        irm = target_mag / target_mag.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        mixture_phase_estimate = multiply_real_mask(target_mag / mix_mag[:, None], mixture)
        estimates = {
            "mixture_baseline": mixture[:, None].expand_as(targets),
            "ideal_ratio_mask": multiply_real_mask(irm, mixture),
            "oracle_phase": targets,
            "ideal_complex": targets,
            "mixture_phase_target_magnitude": mixture_phase_estimate,
        }
        target_wave = istft(targets, metadata["n_fft"], metadata["hop_length"], metadata["chunk_samples"])
        for name, estimate in estimates.items():
            estimate_wave = istft(estimate, metadata["n_fft"], metadata["hop_length"], metadata["chunk_samples"])
            totals[f"{name}_si_sdr"] += (si_sdr(estimate_wave, target_wave) * activity).sum(dim=0)
            totals[f"{name}_snr"] += (snr(estimate_wave, target_wave) * activity).sum(dim=0)
        counts += activity.sum(dim=0)
    per_source = {}
    for index, source in enumerate(dataset.source_names):
        per_source[source] = {key: float(value[index] / counts[index].clamp_min(1)) for key, value in totals.items()}
        per_source[source]["active_chunks"] = int(counts[index])
    valid = counts > 0
    global_metrics = {
        key: float((value[valid] / counts[valid]).mean()) for key, value in totals.items()
    }
    report = {
        "data_dir": str(args.data_dir.resolve()),
        "evaluated_active_chunks": int(counts.sum()),
        "global": global_metrics,
        "per_source": per_source,
        "note": "oracle_phase and ideal_complex are identity ceilings and verify metric/reconstruction correctness.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
