from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch

from .audio import istft, normalize_example
from .config import ExperimentConfig
from .losses import separation_loss
from .metrics import si_sdr, snr
from .runtime import reduce_sum


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = defaultdict(float)
    seen = 0
    accumulation = max(1, config.train.accumulation_steps)
    for batch_index, batch in enumerate(loader):
        if config.train.max_train_batches is not None and batch_index >= config.train.max_train_batches:
            break
        mix = batch["mix"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        activity = batch["activity"].to(device, non_blocking=True)
        mix, targets, _ = normalize_example(mix, targets)
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        sync_context = nullcontext()
        if hasattr(model, "no_sync") and not should_step:
            sync_context = model.no_sync()
        with sync_context, _autocast(device, config.train.amp):
            output = model(mix)
            loss, terms = separation_loss(
                output,
                mix,
                targets,
                activity,
                config.loss,
                config.data.n_fft,
                config.data.hop_length,
                config.data.chunk_samples,
            )
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        batch_size = mix.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        for key, value in terms.items():
            totals[key] += float(value.detach()) * batch_size
        seen += batch_size
    return {key: value / max(1, seen) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: ExperimentConfig,
    num_sources: int,
) -> dict[str, Any]:
    model.eval()
    scalar_names = [
        "loss", "complex_l1", "log_magnitude", "spectral_convergence",
        "waveform_l1", "si_sdr_loss", "activity", "mixture_consistency",
    ]
    sums = torch.zeros(len(scalar_names), device=device, dtype=torch.float64)
    examples = torch.zeros(1, device=device, dtype=torch.float64)
    sisdr_sum = torch.zeros(num_sources, device=device, dtype=torch.float64)
    snr_sum = torch.zeros(num_sources, device=device, dtype=torch.float64)
    active_count = torch.zeros(num_sources, device=device, dtype=torch.float64)
    activity_correct = torch.zeros(1, device=device, dtype=torch.float64)
    activity_total = torch.zeros(1, device=device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if config.train.max_val_batches is not None and batch_index >= config.train.max_val_batches:
            break
        mix = batch["mix"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        activity = batch["activity"].to(device, non_blocking=True)
        mix, targets, _ = normalize_example(mix, targets)
        with _autocast(device, config.train.amp):
            output = model(mix)
            loss, terms = separation_loss(
                output, mix, targets, activity, config.loss,
                config.data.n_fft, config.data.hop_length, config.data.chunk_samples,
            )
        batch_size = mix.shape[0]
        values = {"loss": loss, **terms}
        for index, name in enumerate(scalar_names):
            sums[index] += values[name].double() * batch_size
        examples += batch_size
        estimate_wave = istft(output["estimates"].float(), config.data.n_fft, config.data.hop_length, config.data.chunk_samples)
        target_wave = istft(targets.float(), config.data.n_fft, config.data.hop_length, config.data.chunk_samples)
        source_sisdr = si_sdr(estimate_wave, target_wave)
        source_snr = snr(estimate_wave, target_wave)
        sisdr_sum += (source_sisdr * activity).sum(dim=0).double()
        snr_sum += (source_snr * activity).sum(dim=0).double()
        active_count += activity.sum(dim=0).double()
        predicted_activity = output["activity_logits"] > 0
        activity_correct += (predicted_activity == activity.bool()).sum()
        activity_total += activity.numel()
    for tensor in (sums, examples, sisdr_sum, snr_sum, active_count, activity_correct, activity_total):
        reduce_sum(tensor)
    result = {name: float(sums[i] / examples.clamp_min(1)) for i, name in enumerate(scalar_names)}
    valid = active_count > 0
    per_source_sisdr = sisdr_sum / active_count.clamp_min(1)
    per_source_snr = snr_sum / active_count.clamp_min(1)
    result["si_sdr"] = float(per_source_sisdr[valid].mean()) if valid.any() else float("nan")
    result["snr"] = float(per_source_snr[valid].mean()) if valid.any() else float("nan")
    result["activity_accuracy"] = float(activity_correct / activity_total.clamp_min(1))
    result["per_source_si_sdr"] = per_source_sisdr.cpu().tolist()
    result["per_source_snr"] = per_source_snr.cpu().tolist()
    result["per_source_active_chunks"] = active_count.cpu().tolist()
    return result
