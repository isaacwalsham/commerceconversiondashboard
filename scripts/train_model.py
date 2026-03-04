#!/usr/bin/env python3
"""Train conversion model and save reports."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from commerceconversiondashboard.train import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conversion prediction model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to local online_shoppers_intention.csv",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download from UCI if no --data-path is provided.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_training_pipeline(
        data_path=Path(args.data_path) if args.data_path else None,
        force_download=args.force_download,
    )
    print(json.dumps(summary.__dict__, indent=2))
