"""Train the BBB permeability classifier with active learning.

Usage
-----
    python scripts/train_bbb.py --config config/bbb_predictor.yaml

Implements the BBB prediction model from Section 2.5:
  AttentiveFP + margin sampling active learning
  Target: MCC = 0.8215 at 918 labeled molecules (42 query rounds)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from galaxyair.bbb.train import train_with_active_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BBB permeability classifier with active learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/bbb_predictor.yaml",
        help="Path to bbb_predictor.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model weights and history. "
             "Defaults to paths.weights_dir in the config.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    model = train_with_active_learning(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(f"Training complete. Model: {model}")


if __name__ == "__main__":
    main()
