from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


def mel_band_edges(n_freqs: int, num_bands: int, sample_rate: int, n_fft: int) -> list[int]:
    if num_bands > n_freqs:
        raise ValueError("num_bands cannot exceed the number of frequency bins")
    max_hz = sample_rate / 2
    max_mel = 2595.0 * math.log10(1.0 + max_hz / 700.0)
    raw = []
    for i in range(num_bands + 1):
        mel = max_mel * i / num_bands
        hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        raw.append(round(hz / max_hz * (n_freqs - 1)))
    edges = [0]
    for i in range(1, num_bands):
        minimum = edges[-1] + 1
        maximum = n_freqs - (num_bands - i)
        edges.append(min(max(raw[i], minimum), maximum))
    edges.append(n_freqs)
    return edges


def sinusoidal_position(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class AxialTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        kwargs = dict(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time = nn.TransformerEncoderLayer(**kwargs)
        self.band = nn.TransformerEncoderLayer(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, bands, frames, dim = x.shape
        temporal = x.reshape(batch * bands, frames, dim)
        temporal = self.time(temporal).reshape(batch, bands, frames, dim)
        spectral = temporal.permute(0, 2, 1, 3).reshape(batch * frames, bands, dim)
        spectral = self.band(spectral).reshape(batch, frames, bands, dim)
        return spectral.permute(0, 2, 1, 3)


class AudioPrism(nn.Module):
    """Band-split axial transformer predicting source-conditioned complex ratio masks."""

    def __init__(
        self,
        n_freqs: int,
        num_sources: int,
        sample_rate: int,
        n_fft: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        self.num_sources = num_sources
        self.config = config
        self.edges = mel_band_edges(n_freqs, config.num_bands, sample_rate, n_fft)
        widths = [right - left for left, right in zip(self.edges[:-1], self.edges[1:])]
        self.band_encoders = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, config.dim)) for width in widths
        )
        self.band_embedding = nn.Parameter(torch.randn(config.num_bands, config.dim) * 0.02)
        self.source_embedding = nn.Parameter(torch.randn(num_sources, config.dim) * 0.02)
        self.blocks = nn.ModuleList(
            AxialTransformerBlock(config.dim, config.heads, config.ff_mult, config.dropout)
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.dim)
        self.mask_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.dim),
                nn.Linear(config.dim, config.dim * 2),
                nn.GELU(),
                nn.Linear(config.dim * 2, 2 * width),
            )
            for width in widths
        )
        self.activity_head = nn.Sequential(
            nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim), nn.GELU(), nn.Linear(config.dim, 1)
        )

    def _encode(self, mixture: torch.Tensor) -> torch.Tensor:
        features = []
        for encoder, (left, right) in zip(self.band_encoders, zip(self.edges[:-1], self.edges[1:])):
            band = mixture[:, :, left:right, :].permute(0, 3, 1, 2).flatten(2)
            features.append(encoder(band))
        x = torch.stack(features, dim=1)
        x = x + self.band_embedding[None, :, None, :]
        x = x + sinusoidal_position(x.shape[2], x.shape[3], x.device, x.dtype)[None, None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    @staticmethod
    def _complex_multiply(mask: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        mix_real, mix_imag = mixture[:, None, 0], mixture[:, None, 1]
        mask_real, mask_imag = mask[:, :, 0], mask[:, :, 1]
        return torch.stack(
            (mask_real * mix_real - mask_imag * mix_imag, mask_real * mix_imag + mask_imag * mix_real),
            dim=2,
        )

    @staticmethod
    def project_mixture_consistency(estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        residual = mixture - estimates.sum(dim=1)
        return estimates + residual[:, None] / estimates.shape[1]

    def forward(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        if mixture.ndim != 4 or mixture.shape[1] != 2:
            raise ValueError(f"Expected mixture [batch, 2, freq, time], got {tuple(mixture.shape)}")
        x = self._encode(mixture)
        conditioned = x[:, None] + self.source_embedding[None, :, None, None, :]
        mask = mixture.new_zeros(mixture.shape[0], self.num_sources, 2, self.n_freqs, mixture.shape[-1])
        for band_index, (head, (left, right)) in enumerate(
            zip(self.mask_heads, zip(self.edges[:-1], self.edges[1:]))
        ):
            band_mask = head(conditioned[:, :, band_index]).tanh() * self.config.mask_scale
            band_mask = band_mask.view(
                mixture.shape[0], self.num_sources, mixture.shape[-1], 2, right - left
            ).permute(0, 1, 3, 4, 2)
            mask[:, :, :, left:right, :] = band_mask
        estimates = self._complex_multiply(mask, mixture)
        if self.config.mixture_consistency:
            estimates = self.project_mixture_consistency(estimates, mixture)
        pooled = x.mean(dim=(1, 2))[:, None] + self.source_embedding[None]
        activity_logits = self.activity_head(pooled).squeeze(-1)
        return {"estimates": estimates, "mask": mask, "activity_logits": activity_logits}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
