"""
Training pipeline for the BBB permeability classifier.

Implements Section 2.5 of the paper:
  - AttentiveFP graph neural network
  - Active learning with margin sampling (best) or entropy sampling
  - Final model: MCC = 0.8215, 918 labeled molecules, 42 query rounds
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from galaxyair.bbb.active_learning import ActiveLearner
from galaxyair.bbb.model import (
    BBBClassifier,
    MolecularGraphDataset,
    _collate_graphs,
    evaluate,
    train_epoch,
)

logger = logging.getLogger(__name__)


def load_bbb_dataset(
    csv_path: str | Path,
    smiles_col: str = "SMILES",
    label_col: str = "BBB",
) -> Tuple[List[str], List[int]]:
    """Load SMILES and binary BBB labels from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    smiles_col:
        Column name containing SMILES strings.
    label_col:
        Column name containing binary labels (1 = BBB+, 0 = BBB-).

    Returns
    -------
    (smiles_list, labels)
    """
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].tolist()
    labels = df[label_col].astype(int).tolist()
    return smiles, labels


def train_with_active_learning(
    config_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> BBBClassifier:
    """Full training pipeline with active learning.

    Parameters
    ----------
    config_path:
        Path to bbb_predictor.yaml config file.
    output_dir:
        Directory to save model weights and training history.
        Defaults to the weights_dir specified in the config.

    Returns
    -------
    Trained BBBClassifier.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ---- Load data ----
    smiles, labels = load_bbb_dataset(
        cfg["paths"]["data"],
        smiles_col=cfg["paths"]["smiles_col"],
        label_col=cfg["paths"]["label_col"],
    )
    logger.info(f"Loaded {len(smiles)} molecules  "
                f"(BBB+: {sum(labels)}, BBB-: {len(labels) - sum(labels)})")

    # ---- Train/val split ----
    (train_smi, val_smi,
     train_lbl, val_lbl) = train_test_split(
        smiles, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # ---- Initial labeled / unlabeled split for AL ----
    al_cfg = cfg["active_learning"]
    initial_size = al_cfg["initial_labeled_size"]
    (init_smi, pool_smi,
     init_lbl, pool_lbl) = train_test_split(
        train_smi, train_lbl,
        train_size=initial_size,
        stratify=train_lbl,
        random_state=42,
    )

    # ---- Initialize model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    feat_cfg = cfg["featurizer"]
    model = BBBClassifier(
        node_feat_size=feat_cfg["atom_feat_size"],
        edge_feat_size=feat_cfg["bond_feat_size"],
        graph_feat_size=model_cfg["graph_feat_size"],
        num_layers=model_cfg["num_layers"],
        num_timesteps=model_cfg["num_timesteps"],
        dropout=model_cfg["dropout"],
        device=device,
    )

    # ---- Active learning ----
    learner = ActiveLearner(
        model=model,
        strategy=al_cfg["strategy"],
        n_queries_per_round=al_cfg["n_queries_per_round"],
        max_rounds=al_cfg["max_rounds"],
    )
    history = learner.fit(
        labeled_smiles=init_smi,
        labeled_labels=init_lbl,
        unlabeled_smiles=pool_smi,
        unlabeled_labels=pool_lbl,
        epochs_per_round=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        val_smiles=val_smi,
        val_labels=val_lbl,
    )

    # ---- Save model and history ----
    save_dir = Path(output_dir or cfg["paths"]["weights_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    weights_path = save_dir / "bbb_classifier.pt"
    model.save(weights_path)
    logger.info(f"Model saved to {weights_path}")

    df_history = pd.DataFrame(history)
    history_path = save_dir / "al_history.csv"
    df_history.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")

    # ---- Final validation metrics ----
    val_dataset = MolecularGraphDataset(val_smi, val_lbl)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=_collate_graphs
    )
    val_acc, val_mcc = evaluate(model, val_loader)
    logger.info(f"Final validation — accuracy: {val_acc:.4f}  MCC: {val_mcc:.4f}")

    return model
