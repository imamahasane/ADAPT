from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def build_logger(log_dir: str | Path, name: str = "adapt") -> tuple[logging.Logger, SummaryWriter, JsonlLogger]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    writer = SummaryWriter(log_dir=str(log_dir / "tb"))
    jsonl = JsonlLogger(log_dir / "metrics.jsonl")
    return logger, writer, jsonl
