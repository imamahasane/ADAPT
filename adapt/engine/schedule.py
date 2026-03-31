from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


def current_stage(epoch: int, stage1_end: int = 20, stage2_end: int = 80) -> int:
    if epoch <= stage1_end:
        return 1
    if epoch <= stage2_end:
        return 2
    return 3


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vfe.backbone"):
            backbone_params.append(param)
        else:
            other_params.append(param)
    param_groups = [
        {"params": other_params, "lr": cfg["train"]["lr_main"], "weight_decay": 0.0},
        {"params": backbone_params, "lr": cfg["train"]["lr_backbone"], "weight_decay": 0.0},
    ]
    return torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)


def apply_stage_lr(optimizer: torch.optim.Optimizer, stage: int, epoch: int, cfg: Dict) -> None:
    main_lr = cfg["train"]["lr_main"]
    backbone_lr = 0.0
    if stage == 2:
        backbone_lr = cfg["train"]["lr_backbone_stage2"]
    elif stage == 3:
        backbone_lr = cfg["train"]["lr_backbone_stage3"]
    if epoch >= cfg["train"]["decay_epoch"]:
        main_lr *= cfg["train"]["lr_decay_factor"]
        backbone_lr *= cfg["train"]["lr_decay_factor"]
    optimizer.param_groups[0]["lr"] = main_lr
    optimizer.param_groups[1]["lr"] = backbone_lr
