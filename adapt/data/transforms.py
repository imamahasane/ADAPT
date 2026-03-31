from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass
class ClipAugmenter:
    horizontal_flip_p: float = 0.5
    train: bool = True

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.train:
            return sample
        do_flip = np.random.rand() < self.horizontal_flip_p
        if not do_flip:
            return sample

        for key in ["rgb", "local_depth", "global_semantic", "global_depth"]:
            sample[key] = torch.flip(sample[key], dims=[-1])

        width = 256.0
        bboxes = sample["bbox"].clone()
        x1 = bboxes[..., 0].clone()
        x2 = bboxes[..., 2].clone()
        bboxes[..., 0] = 1.0 - x2
        bboxes[..., 2] = 1.0 - x1
        sample["bbox"] = bboxes

        pose = sample["pose"].clone()
        if pose.shape[-1] == 36:
            coords = pose[..., :34].reshape(*pose.shape[:-1], 17, 2)
            coords[..., 0] = 1.0 - coords[..., 0]
            pose[..., :34] = coords.reshape(*pose.shape[:-1], 34)
        sample["pose"] = pose
        return sample


def normalize_rgb_like(t: torch.Tensor) -> torch.Tensor:
    return (t - IMAGENET_MEAN.to(t.device)) / IMAGENET_STD.to(t.device)
