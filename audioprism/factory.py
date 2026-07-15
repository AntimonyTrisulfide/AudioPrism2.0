from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig, _merge_dataclass
from .model import AudioPrism
from .runtime import load_checkpoint


def build_model(config: ExperimentConfig, num_sources: int) -> AudioPrism:
    return AudioPrism(
        n_freqs=config.data.n_fft // 2 + 1,
        num_sources=num_sources,
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        config=config.model,
    )


def model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[AudioPrism, ExperimentConfig, list[str], dict]:
    checkpoint = load_checkpoint(path, device)
    config = _merge_dataclass(ExperimentConfig(), checkpoint["config"])
    source_names = list(checkpoint["source_names"])
    model = build_model(config, len(source_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, source_names, checkpoint
