from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover
    raise ImportError("mamba-ssm>=1.2.0 is required for the exact MFE implementation.") from exc


class MotionFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 41,
        model_dim: int = 256,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim, bias=False)
        self.embed_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList(
            [Mamba(d_model=model_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, speed: torch.Tensor, bbox: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        motion = torch.cat([speed, bbox, pose], dim=-1)
        x = self.embed_norm(self.embed(motion))
        for layer in self.layers:
            x = layer(x) + x
        return self.out_norm(x)
