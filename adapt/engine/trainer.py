from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from adapt.data.datasets import MultiModalClipDataset, collate_batch
from adapt.data.sampler import build_weighted_sampler
from adapt.engine.evaluator import evaluate
from adapt.engine.schedule import apply_stage_lr, build_optimizer, current_stage
from adapt.losses.intent_loss import CompositeIntentLoss
from adapt.models.adapt import ADAPTModel
from adapt.utils.checkpoint import save_checkpoint
from adapt.utils.distributed import (
    ddp_model,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    maybe_autocast,
    sync_bn_if_needed,
    cleanup_distributed,
)
from adapt.utils.logging import build_logger
from adapt.utils.seed import seed_everything


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("-inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_loaders(cfg: Dict, distributed: bool):
    train_ds = MultiModalClipDataset(
        manifest_path=cfg["data"]["train_manifest"],
        train=True,
        image_size=cfg["data"]["image_size"],
        expected_frames=cfg["data"]["observation_len"],
        speed_max=cfg["data"]["speed_max"],
    )
    val_ds = MultiModalClipDataset(
        manifest_path=cfg["data"]["val_manifest"],
        train=False,
        image_size=cfg["data"]["image_size"],
        expected_frames=cfg["data"]["observation_len"],
        speed_max=cfg["data"]["speed_max"],
    )
    test_ds = MultiModalClipDataset(
        manifest_path=cfg["data"]["test_manifest"],
        train=False,
        image_size=cfg["data"]["image_size"],
        expected_frames=cfg["data"]["observation_len"],
        speed_max=cfg["data"]["speed_max"],
    )

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
    else:
        train_sampler = build_weighted_sampler(train_ds.sample_weights())

    val_sampler = None
    test_sampler = None

    common = dict(num_workers=cfg["data"]["num_workers"], pin_memory=True, collate_fn=collate_batch)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size_per_gpu"], sampler=train_sampler, shuffle=False, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size_per_gpu"], sampler=val_sampler, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size_per_gpu"], sampler=test_sampler, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds.class_weights()


def train(cfg: Dict) -> None:
    distributed, rank, world_size, local_rank = init_distributed(cfg["dist"]["backend"])
    seed_everything(cfg["seed"] + rank, deterministic=cfg["reproducibility"]["deterministic"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log_dir = Path(cfg["output"]["root"]) / cfg["experiment_name"]
    logger, writer, jsonl = build_logger(log_dir)
    logger.info("Starting training: %s", cfg["experiment_name"])
    logger.info("Distributed=%s rank=%s world_size=%s", distributed, rank, world_size)

    train_loader, val_loader, test_loader, class_weights = build_loaders(cfg, distributed)
    model = ADAPTModel(cfg)
    model.set_finetune_stage(1)
    model = sync_bn_if_needed(model, cfg["dist"]["sync_batchnorm"])
    model.to(device)
    model = ddp_model(model, distributed, local_rank)
    optimizer = build_optimizer(model.module if hasattr(model, "module") else model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and torch.cuda.is_available())
    prev_stage = None
    criterion = CompositeIntentLoss(mu=cfg["train"]["loss"]["mu"])
    early_stopper = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])

    best_state = None
    best_val_auc = float("-inf")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        stage = current_stage(epoch, cfg["train"]["schedule"]["stage1_end"], cfg["train"]["schedule"]["stage2_end"])
        base_model = model.module if hasattr(model, "module") else model
        if stage != prev_stage:
            base_model.set_finetune_stage(stage)
            optimizer = build_optimizer(base_model, cfg)
            prev_stage = stage
        apply_stage_lr(optimizer, stage, epoch, cfg)

        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main_process())
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            sample_weight = torch.tensor(
                [class_weights[int(v.item())] for v in batch["label"]], device=device, dtype=batch["label"].dtype
            )
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(cfg["train"]["amp"]):
                out = model(batch)
                loss_dict = criterion(
                    prob=out["prob"],
                    target=batch["label"],
                    sample_weight=sample_weight,
                    head_weights=base_model.tff.head_weight_matrices(),
                )
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss_dict["loss"].detach().cpu())
            pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, amp=cfg["train"]["amp"], threshold=cfg["eval"]["threshold"])
        logger.info("Epoch %03d | stage=%d | train_loss=%.4f | val=%s", epoch, stage, train_loss, val_metrics)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        jsonl.log({"epoch": epoch, "stage": stage, "train_loss": train_loss, **val_metrics})

        checkpoint_state = {
            "epoch": epoch,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        save_checkpoint(checkpoint_state, log_dir / "last.pt")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(checkpoint_state)
            save_checkpoint(best_state, log_dir / "best.pt")

        if early_stopper.step(val_metrics["auc"]):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")

    base_model.load_state_dict(best_state["model"])
    test_metrics = evaluate(model, test_loader, device, amp=cfg["train"]["amp"], threshold=cfg["eval"]["threshold"])
    logger.info("Test metrics: %s", test_metrics)
    jsonl.log({"split": "test", **test_metrics})
    writer.close()
    cleanup_distributed()
