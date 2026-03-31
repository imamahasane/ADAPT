from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from adapt.data.manifest import load_manifest
from adapt.data.transforms import ClipAugmenter, normalize_rgb_like


class MultiModalClipDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        train: bool,
        image_size: int = 256,
        expected_frames: int = 16,
        speed_max: float = 60.0,
    ) -> None:
        self.samples: List[Dict[str, Any]] = load_manifest(manifest_path)
        self.train = train
        self.image_size = image_size
        self.expected_frames = expected_frames
        self.speed_max = speed_max
        self.augment = ClipAugmenter(horizontal_flip_p=0.5, train=train)
        self.class_counts = self._compute_class_counts()

    def _compute_class_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for sample in self.samples:
            counts[int(sample["label"])] += 1
        return counts

    def class_weights(self) -> Dict[int, float]:
        total = len(self.samples)
        return {c: total / (2.0 * max(n, 1)) for c, n in self.class_counts.items()}

    def sample_weights(self) -> List[float]:
        weights = self.class_weights()
        return [weights[int(s["label"])] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str | Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return normalize_rgb_like(tensor)

    def _stack_images(self, paths: List[str]) -> torch.Tensor:
        if len(paths) != self.expected_frames:
            raise ValueError(f"Expected {self.expected_frames} frames, got {len(paths)}")
        return torch.stack([self._load_image(p) for p in paths], dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        out = {
            "rgb": self._stack_images(sample["rgb_paths"]),
            "local_depth": self._stack_images(sample["local_depth_paths"]),
            "global_semantic": self._stack_images(sample["global_semantic_paths"]),
            "global_depth": self._stack_images(sample["global_depth_paths"]),
            "speed": torch.tensor(sample["speed"], dtype=torch.float32).view(self.expected_frames, 1),
            "bbox": torch.tensor(sample["bbox"], dtype=torch.float32).view(self.expected_frames, 4),
            "pose": torch.tensor(sample["pose"], dtype=torch.float32).view(self.expected_frames, -1),
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
            "sample_id": sample.get("sample_id", str(index)),
            "dataset": sample.get("dataset", "unknown"),
            "tte": torch.tensor(sample.get("tte", -1.0), dtype=torch.float32),
        }
        out["speed"] = torch.clamp(out["speed"], min=0.0, max=self.speed_max) / self.speed_max
        out["bbox"] = torch.clamp(out["bbox"], min=0.0, max=float(self.image_size)) / float(self.image_size)
        out = self.augment(out)
        return out


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys_to_stack = ["rgb", "local_depth", "global_semantic", "global_depth", "speed", "bbox", "pose", "label", "tte"]
    out: Dict[str, Any] = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_to_stack}
    out["sample_id"] = [b["sample_id"] for b in batch]
    out["dataset"] = [b["dataset"] for b in batch]
    return out
