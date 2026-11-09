#!/usr/bin/env python3

from src.inference_s3diff_tile import main
from src.my_utils.testing_utils import parse_args_paired_testing

if __name__ == "__main__":
    main(parse_args_paired_testing())
