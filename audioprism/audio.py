from __future__ import annotations

import torch


def channels_to_complex(stft: torch.Tensor) -> torch.Tensor:
    return torch.complex(stft[..., 0, :, :], stft[..., 1, :, :])


def complex_to_channels(stft: torch.Tensor) -> torch.Tensor:
    return torch.stack((stft.real, stft.imag), dim=-3)


def istft(stft: torch.Tensor, n_fft: int, hop_length: int, length: int) -> torch.Tensor:
    shape = stft.shape
    flat = channels_to_complex(stft).reshape(-1, shape[-2], shape[-1])
    window = torch.hann_window(n_fft, device=stft.device, dtype=stft.dtype)
    waveform = torch.istft(flat, n_fft=n_fft, hop_length=hop_length, window=window, length=length)
    return waveform.reshape(*shape[:-3], length)


def stft(waveform: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    value = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    return complex_to_channels(value)


def normalize_example(mix: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    scale = mix.square().sum(dim=1).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-5)
    normalized_mix = mix / scale[:, None]
    normalized_targets = None if targets is None else targets / scale[:, None, None]
    return normalized_mix, normalized_targets, scale
