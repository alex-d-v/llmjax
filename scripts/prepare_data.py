#!/usr/bin/env python3
"""Download and tokenize dataset."""

import argparse
import yaml
from src.data import prepare_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prepare_dataset(cfg, output_dir="data")


if __name__ == "__main__":
    main()