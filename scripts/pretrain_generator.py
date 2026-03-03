"""Pretrain the SMILES VAE on the KRAS triplet dataset.

Usage
-----
    python scripts/pretrain_generator.py --config config/generator.yaml

This script runs contrastive pretraining of the SmilesAutoencoder on
(Lead, Opt, Control) triplets derived from ChEMBL KRAS analogs.

Input
-----
  paths.train_data  : space-separated triplet file (src tar neg per line)
  paths.valid_data  : one SMILES per line (validation set)
  paths.vocab       : path to save/load vocabulary CSV

Output
------
  paths.weights_dir/pretrained_weights.pt  : encoder + decoder weights
  paths.vocab                              : vocabulary CSV
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from galaxyair.generator.autoencoder import SmilesAutoencoder
from galaxyair.generator.dataset import TripletSmilesDataset, ValidationSmilesDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the SMILES VAE generator on triplet data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/generator.yaml",
        help="Path to generator.yaml",
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
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Datasets ----
    train_path = cfg["paths"]["train_data"]
    vocab_path = cfg["paths"]["vocab"]

    vocab_file = Path(vocab_path)
    if vocab_file.exists():
        logger.info("Loading existing vocabulary from %s", vocab_path)
        train_dataset = TripletSmilesDataset(train_path, device, vocab_path=vocab_path)
    else:
        logger.info("Building vocabulary from training data …")
        train_dataset = TripletSmilesDataset(train_path, device)
        train_dataset.save_vocab(vocab_path)
        logger.info("Vocabulary saved to %s  (size=%d)", vocab_path, train_dataset.vocab.size)

    val_dataset = ValidationSmilesDataset(
        cfg["paths"]["valid_data"], vocab_path, device
    )

    # ---- Model ----
    model_cfg = cfg["model"]
    pretrain_cfg = cfg["pretraining"]
    model = SmilesAutoencoder(
        vocab=train_dataset.vocab,
        hidden_size=model_cfg["hidden_size"],
        latent_size=model_cfg["latent_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        device=device,
    )

    # ---- Pretrain ----
    model.pretrain(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=pretrain_cfg["epochs"],
        batch_size=pretrain_cfg["batch_size"],
        learning_rate=pretrain_cfg["learning_rate"],
        max_seq_len=pretrain_cfg["max_seq_len"],
        beta_max=pretrain_cfg.get("beta_max", 1.0),
        gamma=pretrain_cfg.get("gamma", 0.5),
    )

    # ---- Save ----
    save_dir = Path(cfg["paths"]["weights_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "pretrained_weights.pt"
    model.save_weights(weights_path)
    logger.info("Pretrained weights saved to %s", weights_path)


if __name__ == "__main__":
    main()
