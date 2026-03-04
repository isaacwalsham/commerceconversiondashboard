#!/usr/bin/env python3
"""Download the UCI Online Shoppers dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from commerceconversiondashboard.data import download_online_shoppers_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UCI online shoppers dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even when local file exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = download_online_shoppers_dataset(force=args.force)
    print(f"Dataset available at: {path}")
