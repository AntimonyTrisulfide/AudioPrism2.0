from __future__ import annotations

import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target
    projection = projection / target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    noise = estimate - projection
    return 10.0 * torch.log10(
        projection.square().sum(dim=-1).clamp_min(eps) / noise.square().sum(dim=-1).clamp_min(eps)
    )


def snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return 10.0 * torch.log10(
        target.square().sum(dim=-1).clamp_min(eps)
        / (target - estimate).square().sum(dim=-1).clamp_min(eps)
    )


def masked_mean(values: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    return (values * activity).sum() / activity.sum().clamp_min(1.0)
