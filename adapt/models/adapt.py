from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from adapt.models.cmga import CMGA
from adapt.models.mfe import MotionFeatureEncoder
from adapt.models.scma import SparseCrossModalAttention
from adapt.models.swin_vfe import SwinVFE
from adapt.models.tff import TemporalFeatureFusion


class ADAPTModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.vfe = SwinVFE(**cfg["model"]["vfe"])
        self.cmga = CMGA(**cfg["model"]["cmga"])
        self.mfe = MotionFeatureEncoder(**cfg["model"]["mfe"])
        self.scma = SparseCrossModalAttention(**cfg["model"]["scma"])
        self.tff = TemporalFeatureFusion(seq_len=cfg["data"]["observation_len"], **cfg["model"]["tff"])

    def set_finetune_stage(self, stage: int) -> None:
        schedule_cfg = self.cfg["train"]["schedule"]
        self.vfe.set_backbone_trainability(stage, strict_vfe_freeze_rule=schedule_cfg.get("strict_vfe_freeze_rule", True))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        visual = self.vfe(batch["rgb"], batch["local_depth"], batch["global_semantic"], batch["global_depth"])
        fl, fg, cmga_aux = self.cmga(visual)
        fm = self.mfe(batch["speed"], batch["bbox"], batch["pose"])
        fscma, scma_aux = self.scma(fl, fg, fm)
        prob, tff_aux = self.tff(fscma)
        return {
            "prob": prob,
            "fl": fl,
            "fg": fg,
            "fm": fm,
            "fscma": fscma,
            "routing_gate": cmga_aux["routing_gate"],
            "scma_scores": scma_aux["scores"],
            "scma_mask": scma_aux["mask"],
            "scma_attention": scma_aux["attention"],
            "cls_repr": tff_aux["cls_repr"],
        }
