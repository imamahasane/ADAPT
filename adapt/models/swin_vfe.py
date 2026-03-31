from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn


class SwinVFE(nn.Module):
    def __init__(
        self,
        model_name: str = "swinv2_tiny_window16_256",
        out_dim: int = 256,
        pretrained: bool = False,
        pretrained_checkpoint: str | None = None,
        strict_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.proj = nn.Sequential(
            nn.Conv2d(768, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        if pretrained_checkpoint:
            self.load_pretrained(Path(pretrained_checkpoint), strict=strict_checkpoint)

    def load_pretrained(self, checkpoint_path: Path, strict: bool = False) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    def forward_modality(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.backbone.forward_features(x)
        if feats.ndim != 4:
            raise RuntimeError(f"Expected 4D backbone feature map, got shape {tuple(feats.shape)}")
        feats = self.proj(feats)
        feats = feats.view(b, n, feats.shape[1], feats.shape[2], feats.shape[3])
        return feats

    def forward(self, rgb: torch.Tensor, local_depth: torch.Tensor, global_semantic: torch.Tensor, global_depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "rgb": self.forward_modality(rgb),
            "local_depth": self.forward_modality(local_depth),
            "global_semantic": self.forward_modality(global_semantic),
            "global_depth": self.forward_modality(global_depth),
        }

    def freeze_patch_and_early_stages(self) -> None:
        frozen_prefixes = ["patch_embed", "layers.0", "layers.1"]
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen_prefixes):
                param.requires_grad = False

    def set_backbone_trainability(self, stage: int, strict_vfe_freeze_rule: bool = True) -> None:
        # Stage 1: freeze all backbone params.
        # Stage 2: unfreeze the last two Swin stages.
        # Stage 3 ambiguity in the paper: one section says all stages unfreeze,
        # another says patch embedding + first two stages stay frozen throughout.
        # strict_vfe_freeze_rule=True follows Sec. 2.2 literally.
        for _, param in self.backbone.named_parameters():
            param.requires_grad = False
        if stage >= 2:
            for name, param in self.backbone.named_parameters():
                if name.startswith("layers.2") or name.startswith("layers.3"):
                    param.requires_grad = True
        if stage >= 3 and not strict_vfe_freeze_rule:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("patch_embed"):
                    param.requires_grad = True
        self.freeze_patch_and_early_stages()
