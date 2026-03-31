from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from adapt.data.datasets import MultiModalClipDataset, collate_batch
from adapt.models.adapt import ADAPTModel
from adapt.utils.checkpoint import load_checkpoint


@torch.no_grad()
def run_inference(config: Dict, checkpoint_path: str | Path, manifest_path: str | Path, output_path: str | Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiModalClipDataset(
        manifest_path=manifest_path,
        train=False,
        image_size=config["data"]["image_size"],
        expected_frames=config["data"]["observation_len"],
        speed_max=config["data"]["speed_max"],
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size_per_gpu"], shuffle=False, collate_fn=collate_batch)
    model = ADAPTModel(config).to(device)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows: List[Dict] = []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        pred = (out["prob"] >= config["eval"]["threshold"]).long().cpu().tolist()
        prob = out["prob"].cpu().tolist()
        mask = out["scma_mask"].cpu().tolist()
        for sid, p, y, m in zip(batch["sample_id"], prob, pred, mask):
            rows.append({"sample_id": sid, "prob": p, "pred": y, "scma_mask": m})

    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
