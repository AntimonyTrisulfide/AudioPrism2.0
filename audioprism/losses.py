from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from .audio import istft
from .config import LossConfig
from .metrics import masked_mean, si_sdr


def separation_loss(
    output: dict[str, torch.Tensor],
    mixture: torch.Tensor,
    targets: torch.Tensor,
    activity: torch.Tensor,
    config: LossConfig,
    n_fft: int,
    hop_length: int,
    chunk_samples: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    estimates = output["estimates"].float()
    targets = targets.float()
    mixture = mixture.float()
    active = activity[:, :, None, None, None]
    active_bins = active.sum().clamp_min(1.0) * targets.shape[-2] * targets.shape[-1] * 2
    complex_l1 = ((estimates - targets).abs() * active).sum() / active_bins

    estimate_mag = estimates.square().sum(dim=2).clamp_min(1.0e-8).sqrt()
    target_mag = targets.square().sum(dim=2).clamp_min(1.0e-8).sqrt()
    active_mag = activity[:, :, None, None]
    mag_bins = active_mag.sum().clamp_min(1.0) * targets.shape[-2] * targets.shape[-1]
    log_magnitude = ((torch.log1p(estimate_mag) - torch.log1p(target_mag)).abs() * active_mag).sum() / mag_bins
    error_energy = ((estimate_mag - target_mag).square() * active_mag).sum()
    target_energy = (target_mag.square() * active_mag).sum()
    spectral_convergence = (
        error_energy.clamp_min(1.0e-12).sqrt()
        / target_energy.clamp_min(1.0e-12).sqrt()
    )

    need_waveform = config.waveform_l1 > 0 or config.si_sdr > 0
    if need_waveform:
        estimate_wave = istft(estimates.float(), n_fft, hop_length, chunk_samples)
        target_wave = istft(targets.float(), n_fft, hop_length, chunk_samples)
        waveform_l1 = masked_mean((estimate_wave - target_wave).abs().mean(dim=-1), activity)
        sisdr_loss = -masked_mean(si_sdr(estimate_wave, target_wave), activity)
    else:
        waveform_l1 = estimates.new_zeros(())
        sisdr_loss = estimates.new_zeros(())

    activity_loss = F.binary_cross_entropy_with_logits(output["activity_logits"].float(), activity.float())
    consistency = (estimates.sum(dim=1) - mixture).abs().mean()
    terms = {
        "complex_l1": complex_l1,
        "log_magnitude": log_magnitude,
        "spectral_convergence": spectral_convergence,
        "waveform_l1": waveform_l1,
        "si_sdr_loss": sisdr_loss,
        "activity": activity_loss,
        "mixture_consistency": consistency,
    }
    weights = asdict(config)
    key_map = {"si_sdr_loss": "si_sdr"}
    total = sum(terms[key] * weights[key_map.get(key, key)] for key in terms)
    return total, terms
