#!/usr/bin/env python3
"""Compatibility entry; training lives in src/train_s3diff.py."""
from src.train_s3diff import main
from src.my_utils.training_utils import parse_args_paired_training

if __name__ == "__main__":
    main(parse_args_paired_training())
