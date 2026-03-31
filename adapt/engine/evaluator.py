from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from adapt.engine.metrics import ClassificationMetrics
from adapt.utils.distributed import maybe_autocast


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, amp: bool, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    metric = ClassificationMetrics(threshold=threshold)
    probs = []
    targets = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with maybe_autocast(amp):
            out = model(batch)
        probs.append(out["prob"].detach().float().cpu().numpy())
        targets.append(batch["label"].detach().float().cpu().numpy())
    prob = np.concatenate(probs)
    target = np.concatenate(targets).astype(np.int64)
    return metric.compute(prob, target)
