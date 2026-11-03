#!/usr/bin/env python3
"""Unified whole-image/tiled restoration test entry, retaining the original filename."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.inference_s3diff_tile import main
from src.my_utils.testing_utils import parse_args_paired_testing

if __name__ == "__main__":
    main(parse_args_paired_testing())
