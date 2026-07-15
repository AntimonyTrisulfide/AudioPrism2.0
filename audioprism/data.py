from __future__ import annotations

import bisect
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def polar_to_channels(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return torch.stack((magnitude * phase.cos(), magnitude * phase.sin()), dim=-3)


class PreprocessedComplexDataset(Dataset):
    """Reads the AudioMask 1.x preprocessed format as complex STFT chunks."""

    def __init__(
        self,
        root: str | Path,
        cache_tracks: int = 2,
        gain_augmentation_db: float = 0.0,
        allow_mixture_phase_targets: bool = False,
    ) -> None:
        self.root = Path(root)
        metadata_path = self.root / "dataset_metadata.pt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
        self.metadata = _load(metadata_path)
        self.source_names = list(self.metadata["all_source_names"])
        self.source_file_map = self.metadata.get("source_filenames", {n: n for n in self.source_names})
        self.track_dirs = sorted(p for p in self.root.iterdir() if p.is_dir() and (p / "mix.pt").exists())
        self.track_metadata = [_load(p / "metadata.pt") for p in self.track_dirs]
        self.cumulative: list[int] = []
        total = 0
        for item in self.track_metadata:
            total += int(item["n_chunks"])
            self.cumulative.append(total)
        self.cache_tracks = max(0, cache_tracks)
        self.gain_augmentation_db = float(gain_augmentation_db)
        self.allow_mixture_phase_targets = allow_mixture_phase_targets
        self._cache: OrderedDict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = OrderedDict()

        required = ("sr", "n_fft", "hop_length", "chunk_samples")
        missing = [key for key in required if key not in self.metadata]
        if missing:
            raise ValueError(f"Dataset metadata is missing: {', '.join(missing)}")

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _resolve(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        track_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if track_index == 0 else self.cumulative[track_index - 1]
        return track_index, index - previous

    def _track(self, track_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if track_index in self._cache:
            value = self._cache.pop(track_index)
            self._cache[track_index] = value
            return value
        track_dir = self.track_dirs[track_index]
        mix = _load(track_dir / "mix.pt")
        sources = [
            _load(track_dir / f"{self.source_file_map.get(name, name)}.pt")
            for name in self.source_names
        ]
        value = (mix, sources)
        if self.cache_tracks:
            self._cache[track_index] = value
            while len(self._cache) > self.cache_tracks:
                self._cache.popitem(last=False)
        return value

    @staticmethod
    def _phase(payload: dict[str, Any], index: int, label: str) -> torch.Tensor:
        phases = payload.get("true_phases", payload.get("phases"))
        if phases is None:
            raise ValueError(f"{label} has no phase data; re-run the phase-aware preprocessor")
        return phases[index].float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        track_index, chunk_index = self._resolve(index)
        mix_payload, source_payloads = self._track(track_index)
        mix_mag = mix_payload["spectrogram"][chunk_index].float()
        mix_phase = self._phase(mix_payload, chunk_index, "mixture")
        mix = polar_to_channels(mix_mag, mix_phase)
        sources = []
        activity = []
        for name, payload in zip(self.source_names, source_payloads):
            magnitude = payload["spectrogram"][chunk_index].float()
            phases = payload.get("true_phases", payload.get("phases"))
            if phases is None:
                if not self.allow_mixture_phase_targets:
                    raise ValueError(f"{name} has no phase data; re-run the phase-aware preprocessor")
                source_phase = mix_phase
            else:
                source_phase = phases[chunk_index].float()
            sources.append(polar_to_channels(magnitude, source_phase))
            activity.append(magnitude.square().mean().sqrt() > 1.0e-5)
        targets = torch.stack(sources)
        if self.gain_augmentation_db > 0:
            gain_db = random.uniform(-self.gain_augmentation_db, self.gain_augmentation_db)
            gain = 10.0 ** (gain_db / 20.0)
            mix = mix * gain
            targets = targets * gain
        return {
            "mix": mix,
            "targets": targets,
            "activity": torch.tensor(activity, dtype=torch.float32),
            "track": self.track_dirs[track_index].name,
            "chunk": chunk_index,
        }


def validate_compatible(train: PreprocessedComplexDataset, other: PreprocessedComplexDataset) -> None:
    keys = ("sr", "n_fft", "hop_length", "chunk_samples")
    for key in keys:
        if train.metadata[key] != other.metadata[key]:
            raise ValueError(f"Dataset mismatch for {key}: {train.metadata[key]} != {other.metadata[key]}")
    if train.source_names != other.source_names:
        raise ValueError("Training and evaluation source ordering differs")


def dataset_summary(dataset: PreprocessedComplexDataset) -> dict[str, Any]:
    return {
        "root": str(dataset.root.resolve()),
        "tracks": len(dataset.track_dirs),
        "chunks": len(dataset),
        "sources": dataset.source_names,
        "sample_rate": dataset.metadata["sr"],
        "n_fft": dataset.metadata["n_fft"],
        "hop_length": dataset.metadata["hop_length"],
        "chunk_samples": dataset.metadata["chunk_samples"],
    }
