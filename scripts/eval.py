from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from adapt.config import load_config
from adapt.data.datasets import MultiModalClipDataset, collate_batch
from adapt.engine.evaluator import evaluate
from adapt.models.adapt import ADAPTModel
from adapt.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = cfg["data"][f"{args.split}_manifest"]
    dataset = MultiModalClipDataset(
        manifest_path=manifest,
        train=False,
        image_size=cfg["data"]["image_size"],
        expected_frames=cfg["data"]["observation_len"],
        speed_max=cfg["data"]["speed_max"],
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size_per_gpu"], shuffle=False, collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ADAPTModel(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    metrics = evaluate(model, loader, device, amp=cfg["train"]["amp"], threshold=cfg["eval"]["threshold"])
    print(metrics)


if __name__ == "__main__":
    main()
