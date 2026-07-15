from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio

from audioprism.audio import istft, normalize_example, stft
from audioprism.factory import model_from_checkpoint


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "source"


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    try:
        waveform, original_rate = torchaudio.load(path)
    except (ImportError, RuntimeError):
        try:
            import soundfile as sf

            samples, original_rate = sf.read(path, dtype="float32", always_2d=True)
        except ImportError:
            import numpy as np
            from scipy.io import wavfile

            original_rate, samples = wavfile.read(path)
            if np.issubdtype(samples.dtype, np.integer):
                samples = samples.astype("float32") / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
            else:
                samples = samples.astype("float32")
            if samples.ndim == 1:
                samples = samples[:, None]
        waveform = torch.from_numpy(samples.T.copy())
    waveform = waveform.mean(dim=0)
    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    return waveform


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    try:
        torchaudio.save(path, waveform, sample_rate)
    except (ImportError, RuntimeError):
        try:
            import soundfile as sf

            sf.write(path, waveform.squeeze(0).numpy(), sample_rate)
        except ImportError:
            from scipy.io import wavfile

            wavfile.write(path, sample_rate, waveform.squeeze(0).numpy().astype("float32"))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Separate an audio file with AudioPrism 2.0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--activity-threshold", type=float, default=0.25)
    parser.add_argument("--max-seconds", type=float, help="Optional duration limit for a quick inference check")
    args = parser.parse_args()
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, source_names, checkpoint = model_from_checkpoint(args.checkpoint, device)
    waveform = load_mono(args.input, config.data.sample_rate)
    if args.max_seconds is not None:
        waveform = waveform[: int(args.max_seconds * config.data.sample_rate)]
    original_length = waveform.numel()
    chunk = config.data.chunk_samples
    stride = max(1, round(chunk * (1.0 - args.overlap)))
    if original_length <= chunk:
        positions = [0]
    else:
        positions = list(range(0, original_length - chunk + 1, stride))
        final_position = original_length - chunk
        if positions[-1] != final_position:
            positions.append(final_position)
    total_length = max(original_length, positions[-1] + chunk)
    estimates = torch.zeros(len(source_names), total_length)
    weights = torch.zeros(total_length)
    window = torch.hann_window(chunk, periodic=False).sqrt().clamp_min(1.0e-3)
    activity_sum = torch.zeros(len(source_names))
    for start in positions:
        segment = torch.zeros(chunk)
        available = min(chunk, original_length - start)
        if available > 0:
            segment[:available] = waveform[start : start + available]
        mixture_stft = stft(segment[None].to(device), config.data.n_fft, config.data.hop_length)
        normalized, _, scale = normalize_example(mixture_stft)
        output = model(normalized)
        source_stft = output["estimates"] * scale[:, None, None]
        source_wave = istft(
            source_stft.float(), config.data.n_fft, config.data.hop_length, chunk
        )[0].cpu()
        probabilities = output["activity_logits"].sigmoid()[0].cpu()
        source_wave[probabilities < args.activity_threshold] = 0
        estimates[:, start : start + chunk] += source_wave * window
        weights[start : start + chunk] += window
        activity_sum += probabilities
    estimates = estimates[:, :original_length] / weights[:original_length].clamp_min(1.0e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in zip(source_names, estimates):
        save_audio(args.output_dir / f"{safe_name(name)}.wav", audio[None], config.data.sample_rate)
    report = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "sample_rate": config.data.sample_rate,
        "duration_seconds": original_length / config.data.sample_rate,
        "chunks": len(positions),
        "mean_activity_probability": {
            name: float(value / len(positions)) for name, value in zip(source_names, activity_sum)
        },
    }
    (args.output_dir / "separation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
