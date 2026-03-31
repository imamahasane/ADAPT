from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class CompositeIntentLoss(nn.Module):
    def __init__(self, mu: float = 1e-3) -> None:
        super().__init__()
        self.mu = mu

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor,
        head_weights: list[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-7
        prob = prob.clamp(min=eps, max=1.0 - eps)
        intent = -(sample_weight * (target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))).mean()
        reg = torch.zeros((), device=prob.device, dtype=prob.dtype)
        for weight in head_weights:
            reg = reg + torch.sum(weight.pow(2))
        total = intent + self.mu * reg
        return {"loss": total, "intent": intent, "reg": reg}
