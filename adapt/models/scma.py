from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class SparseCrossModalAttention(nn.Module):
    def __init__(self, in_dim: int = 256, proj_dim: int = 128, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.q = nn.ModuleList([nn.Linear(in_dim, proj_dim) for _ in range(3)])
        self.k = nn.ModuleList([nn.Linear(in_dim, proj_dim) for _ in range(3)])
        self.v = nn.ModuleList([nn.Linear(in_dim, proj_dim) for _ in range(3)])
        self.scale = proj_dim ** -0.5

    def forward(self, fl: torch.Tensor, fg: torch.Tensor, fm: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        streams = [fl, fg, fm]
        qs = [layer(x) for layer, x in zip(self.q, streams)]
        ks = [layer(x) for layer, x in zip(self.k, streams)]
        vs = [layer(x) for layer, x in zip(self.v, streams)]

        # S_{ij} = <Q_i, K_j> / sqrt(d_p), evaluated per frame token.
        scores = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append((qs[i] * ks[j]).sum(dim=-1) * self.scale)
            scores.append(torch.stack(row, dim=-1))
        scores = torch.stack(scores, dim=-2)  # [B, N, 3, 3]

        topk = torch.topk(scores, k=self.top_k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, topk.indices, True)

        neg_inf = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~mask, neg_inf)
        attn = torch.softmax(masked_scores, dim=-1)

        vstack = torch.stack(vs, dim=-2)  # [B, N, 3, D]
        attended = []
        for i in range(3):
            weights = attn[:, :, i, :].unsqueeze(-1)
            attended.append((weights * vstack).sum(dim=-2))
        fused = torch.cat(attended, dim=-1)
        aux = {"scores": scores, "mask": mask.to(scores.dtype), "attention": attn}
        return fused, aux
