from __future__ import annotations

from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(weights: list[float]) -> WeightedRandomSampler:
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
