from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from audioprism.config import config_parser, load_config
from audioprism.data import PreprocessedComplexDataset, dataset_summary, validate_compatible
from audioprism.engine import train_epoch, validate
from audioprism.factory import build_model
from audioprism.runtime import (
    append_jsonl, cleanup_distributed, configure_logging, cosine_schedule,
    distributed_setup, load_checkpoint, save_checkpoint, seed_everything,
    seed_worker, unwrap,
)


def parse_args() -> argparse.Namespace:
    parser = config_parser("Train AudioPrism 2.0")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def check_metadata(config, dataset: PreprocessedComplexDataset) -> None:
    expected = {
        "sample_rate": dataset.metadata["sr"],
        "n_fft": dataset.metadata["n_fft"],
        "hop_length": dataset.metadata["hop_length"],
        "chunk_samples": dataset.metadata["chunk_samples"],
    }
    for key, actual in expected.items():
        configured = getattr(config.data, key)
        if configured != actual:
            raise ValueError(f"Config data.{key}={configured}, but dataset contains {actual}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    device, rank, local_rank, world_size = distributed_setup()
    seed_everything(config.train.seed, rank)
    run_dir = config.run_dir
    if rank == 0:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        config.save(run_dir / "config.yaml")
    if dist.is_initialized():
        dist.barrier()
    logger = configure_logging(run_dir / "logs" / "train.log", rank)
    train_data = PreprocessedComplexDataset(
        config.data.train_dir, config.data.cache_tracks, config.data.gain_augmentation_db,
        config.data.allow_mixture_phase_targets,
    )
    val_data = PreprocessedComplexDataset(
        config.data.val_dir, config.data.cache_tracks, 0.0, config.data.allow_mixture_phase_targets
    )
    validate_compatible(train_data, val_data)
    check_metadata(config, train_data)
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=config.train.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_data, shuffle=False) if world_size > 1 else None
    generator = torch.Generator().manual_seed(config.train.seed + rank)
    loader_kwargs = dict(
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.train.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_data, shuffle=train_sampler is None, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, sampler=val_sampler, **loader_kwargs)
    model = build_model(config, len(train_data.source_names)).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay, betas=(0.9, 0.95)
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.train.accumulation_steps))
    scheduler = cosine_schedule(
        optimizer, config.train.warmup_steps, steps_per_epoch * config.train.epochs, config.train.min_lr_ratio
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.amp and device.type == "cuda")
    start_epoch = 0
    best_si_sdr = float("-inf")
    stale_epochs = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        unwrap(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_si_sdr = float(checkpoint.get("best_si_sdr", best_si_sdr))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        logger.info("Resumed %s at epoch %d", args.resume, start_epoch + 1)
    logger.info("Device=%s world_size=%d parameters=%.2fM", device, world_size, unwrap(model).parameter_count() / 1e6)
    logger.info("Train=%s", dataset_summary(train_data))
    logger.info("Validation=%s", dataset_summary(val_data))
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    try:
        for epoch in range(start_epoch, config.train.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            started = time.time()
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
            val_metrics = None
            improved = False
            if (epoch + 1) % config.train.validate_every == 0:
                val_metrics = validate(model, val_loader, device, config, len(train_data.source_names))
                improved = val_metrics["si_sdr"] > best_si_sdr
                if improved:
                    best_si_sdr = val_metrics["si_sdr"]
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            payload = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": val_metrics,
                "best_si_sdr": best_si_sdr,
            }
            checkpoint = {
                "format_version": 2,
                "epoch": epoch,
                "model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_si_sdr": best_si_sdr,
                "stale_epochs": stale_epochs,
                "config": config.to_dict(),
                "source_names": train_data.source_names,
            }
            if rank == 0:
                append_jsonl(metrics_path, payload)
                save_checkpoint(run_dir / "checkpoints" / "latest.pt", checkpoint)
                if improved:
                    save_checkpoint(run_dir / "checkpoints" / "best.pt", checkpoint)
                if (epoch + 1) % config.train.save_every == 0:
                    save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt", checkpoint)
                logger.info(
                    "Epoch %d/%d train=%.4f val_si_sdr=%s best=%.3f time=%.1fs",
                    epoch + 1, config.train.epochs, train_metrics["loss"],
                    "n/a" if val_metrics is None else f"{val_metrics['si_sdr']:.3f}",
                    best_si_sdr, payload["elapsed_seconds"],
                )
            stop = torch.tensor(
                stale_epochs >= config.train.early_stopping_patience, device=device, dtype=torch.int32
            )
            if dist.is_initialized():
                dist.broadcast(stop, src=0)
            if stop.item():
                logger.info("Early stopping after %d stale validation epochs", stale_epochs)
                break
        if rank == 0 and (run_dir / "checkpoints" / "best.pt").exists():
            shutil.copy2(run_dir / "checkpoints" / "best.pt", run_dir / "audioprism2_best.pt")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
