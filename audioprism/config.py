from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_dir: str = "data/preprocessed_train"
    val_dir: str = "data/preprocessed_val"
    test_dir: str | None = None
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    chunk_samples: int = 128000
    cache_tracks: int = 2
    gain_augmentation_db: float = 6.0
    allow_mixture_phase_targets: bool = False


@dataclass
class ModelConfig:
    num_bands: int = 32
    dim: int = 192
    depth: int = 8
    heads: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    mask_scale: float = 2.0
    mixture_consistency: bool = True


@dataclass
class LossConfig:
    complex_l1: float = 1.0
    log_magnitude: float = 0.5
    spectral_convergence: float = 0.2
    waveform_l1: float = 0.2
    si_sdr: float = 0.1
    activity: float = 0.1
    mixture_consistency: float = 0.05


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 2
    accumulation_steps: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.05
    grad_clip: float = 5.0
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    validate_every: int = 1
    save_every: int = 10
    early_stopping_patience: int = 30
    max_train_batches: int | None = None
    max_val_batches: int | None = None


@dataclass
class ExperimentConfig:
    name: str = "audioprism2_base"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def _parse_scalar(value: str) -> Any:
    return yaml.safe_load(value)


def _apply_override(config: ExperimentConfig, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, received: {expression}")
    path, raw_value = expression.split("=", 1)
    target: Any = config
    keys = path.split(".")
    for key in keys[:-1]:
        if not hasattr(target, key):
            raise ValueError(f"Unknown override path: {path}")
        target = getattr(target, key)
    if not hasattr(target, keys[-1]):
        raise ValueError(f"Unknown override path: {path}")
    setattr(target, keys[-1], _parse_scalar(raw_value))


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = _merge_dataclass(ExperimentConfig(), payload)
    for override in overrides or []:
        _apply_override(config, override)
    return config


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value, e.g. --set train.batch_size=4",
    )
    return parser


def dump_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
