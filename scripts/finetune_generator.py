"""Fine-tune the pretrained SMILES VAE with RL (policy gradient).

Usage
-----
    python scripts/finetune_generator.py --config config/generator.yaml

Reward function (Section 2.4 of paper)
---------------------------------------
    reward = 0.4 × affinity + 0.4 × BBBp + 0.2 × QED

Only decoder parameters are updated during fine-tuning.
The Tanimoto similarity threshold is 0.4.

Input
-----
  paths.weights_dir/pretrained_weights.pt : pretrained encoder+decoder
  paths.finetune_data                     : seed SMILES for generation
  paths.vocab                             : vocabulary CSV

Output
------
  paths.weights_dir/finetuned_weights.pt  : fine-tuned decoder weights
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from galaxyair.affinity.scorer import AffinityScorer
from galaxyair.bbb.scorer import BBBScorer
from galaxyair.generator.autoencoder import SmilesAutoencoder
from galaxyair.generator.dataset import ValidationSmilesDataset
from galaxyair.utils.molecular import compute_qed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the SMILES VAE generator with RL."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/generator.yaml",
        help="Path to generator.yaml",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Override path to pretrained weights (default: from config).",
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

    # ---- Load model ----
    weights_dir = Path(cfg["paths"]["weights_dir"])
    pretrained_path = args.pretrained_weights or str(
        weights_dir / "pretrained_weights.pt"
    )
    model = SmilesAutoencoder.from_config(cfg["paths"]["config_save"], device=device)
    model.load_weights(pretrained_path)
    logger.info("Loaded pretrained weights from %s", pretrained_path)

    # ---- Seed dataset ----
    seed_dataset = ValidationSmilesDataset(
        cfg["paths"]["finetune_data"],
        cfg["paths"]["vocab"],
        device,
    )

    # ---- Scorers ----
    rl_cfg = cfg["rl"]
    affinity_scorer = AffinityScorer(
        weights_path=cfg["paths"].get("surrogate_affinity_weights"),
        device=device,
    )
    bbb_scorer = BBBScorer(
        weights_path=cfg["paths"]["bbb_weights"],
        device=device,
    )

    def scoring_fn(smiles: str) -> float:
        aff = affinity_scorer(smiles)
        bbb = bbb_scorer(smiles)
        qed = compute_qed(smiles)
        return 0.4 * aff + 0.4 * bbb + 0.2 * qed

    finetune_cfg = cfg["finetuning"]

    # ---- Fine-tune ----
    model.finetune(
        seed_dataset=seed_dataset,
        scoring_fn=scoring_fn,
        epochs=finetune_cfg["epochs"],
        batch_size=finetune_cfg["batch_size"],
        learning_rate=finetune_cfg["learning_rate"],
        max_seq_len=finetune_cfg["max_seq_len"],
        n_samples=finetune_cfg.get("n_samples", 128),
        threshold_similarity=rl_cfg["threshold_similarity"],
        threshold_property=rl_cfg.get("threshold_property", 0.0),
        temperature=finetune_cfg.get("temperature", 1.0),
        replay_buffer_size=finetune_cfg.get("replay_buffer_size", 256),
    )

    # ---- Save ----
    finetuned_path = weights_dir / "finetuned_weights.pt"
    model.save_weights(finetuned_path)
    logger.info("Fine-tuned weights saved to %s", finetuned_path)


if __name__ == "__main__":
    main()
