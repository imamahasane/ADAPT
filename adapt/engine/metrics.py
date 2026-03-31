from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    threshold: float = 0.5

    def compute(self, prob: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        pred = (prob >= self.threshold).astype(np.int64)
        return {
            "acc": float(accuracy_score(target, pred)),
            "auc": float(roc_auc_score(target, prob)),
            "f1": float(f1_score(target, pred, zero_division=0)),
            "precision": float(precision_score(target, pred, zero_division=0)),
            "recall": float(recall_score(target, pred, zero_division=0)),
        }
