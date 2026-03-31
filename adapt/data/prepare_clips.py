from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from adapt.data.manifest import save_manifest


def compute_stride(observation_len: int, overlap_ratio: float) -> int:
    return int(observation_len * (1.0 - overlap_ratio))


def build_manifest_stub(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raise NotImplementedError(
        "Error: The build_manifest_stub function is not implemented. Please implement this function to build the manifest stub based on the provided arguments."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-len", type=int, default=16)
    parser.add_argument("--overlap-ratio", type=float, default=0.8)
    parser.add_argument("--tte-min", type=float, default=1.0)
    parser.add_argument("--tte-max", type=float, default=2.0)
    args = parser.parse_args()
    manifest = build_manifest_stub(args)
    save_manifest(manifest, args.output)


if __name__ == "__main__":
    main()
