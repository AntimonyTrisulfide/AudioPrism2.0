from __future__ import annotations

import argparse
import json
from pathlib import Path

from audioprism.data import PreprocessedComplexDataset, dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize an AudioPrism dataset")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--scan-chunks", type=int, default=8)
    parser.add_argument(
        "--allow-mixture-phase-targets", action="store_true",
        help="Permit legacy magnitude-only sources for smoke testing only.",
    )
    args = parser.parse_args()
    dataset = PreprocessedComplexDataset(
        args.data_dir, cache_tracks=1,
        allow_mixture_phase_targets=args.allow_mixture_phase_targets,
    )
    summary = dataset_summary(dataset)
    summary["sample_shapes"] = []
    summary["active_chunk_counts"] = {name: 0 for name in dataset.source_names}
    for index in range(min(args.scan_chunks, len(dataset))):
        sample = dataset[index]
        summary["sample_shapes"].append({
            "mix": list(sample["mix"].shape), "targets": list(sample["targets"].shape)
        })
        for name, active in zip(dataset.source_names, sample["activity"]):
            summary["active_chunk_counts"][name] += int(active)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
