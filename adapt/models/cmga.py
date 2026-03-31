from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ChannelGuidedAttention(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        hidden = channels // reduction
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        avg = guide.mean(dim=(2, 3))
        mx = guide.amax(dim=(2, 3))
        descriptor = avg + mx
        alpha = self.mlp(descriptor).unsqueeze(-1).unsqueeze(-1)
        return target * alpha


class SpatialGuidedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, target: torch.Tensor, guide: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        avg = guide.mean(dim=1, keepdim=True)
        mx = guide.amax(dim=1, keepdim=True)
        descriptor = torch.cat([avg, mx], dim=1)
        beta = self.sigmoid(self.conv(descriptor))
        return target * beta + residual


class FusionPath(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 128) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CMGAPair(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        self.cmgca = ChannelGuidedAttention(channels=channels, reduction=reduction)
        self.cmgsa = SpatialGuidedAttention()
        self.add_path = FusionPath(in_channels=channels, out_channels=channels // 2)
        self.mul_path = FusionPath(in_channels=channels, out_channels=channels // 2)

    def forward(self, fi: torch.Tensor, fj: torch.Tensor) -> torch.Tensor:
        fi_ch = self.cmgca(fi, fj)
        fj_ch = self.cmgca(fj, fi)
        fi_sp = self.cmgsa(fi_ch, fj_ch, fi)
        fj_sp = self.cmgsa(fj_ch, fi_ch, fj)
        fused_add = self.add_path(fi_sp + fj_sp)
        fused_mul = self.mul_path(fi_sp * fj_sp)
        return torch.cat([fused_add, fused_mul], dim=1)


class CMGA(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        self.local_pair = CMGAPair(channels=channels, reduction=reduction)
        self.global_pair = CMGAPair(channels=channels, reduction=reduction)
        self.gate = nn.Linear(channels * 2, 2)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, c, h, w = feats["rgb"].shape
        rgb = feats["rgb"].reshape(b * n, c, h, w)
        ld = feats["local_depth"].reshape(b * n, c, h, w)
        gs = feats["global_semantic"].reshape(b * n, c, h, w)
        gd = feats["global_depth"].reshape(b * n, c, h, w)

        local_map = self.local_pair(rgb, ld)
        global_map = self.global_pair(gs, gd)

        fl = local_map.mean(dim=(2, 3))
        fg = global_map.mean(dim=(2, 3))
        gate = torch.softmax(self.gate(torch.cat([fl, fg], dim=-1)), dim=-1)
        fused_local = self.out_proj(gate[:, :1] * fl + gate[:, 1:] * fg).view(b, n, c)
        fused_global = (fl + fg).view(b, n, c)
        aux = {
            "local_map": local_map.view(b, n, c, h, w),
            "global_map": global_map.view(b, n, c, h, w),
            "routing_gate": gate.view(b, n, 2),
        }
        return fused_local, fused_global, aux
