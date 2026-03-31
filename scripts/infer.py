from __future__ import annotations

import argparse

from adapt.config import load_config
from adapt.engine.infer import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg, args.checkpoint, args.manifest, args.output)


if __name__ == "__main__":
    main()
