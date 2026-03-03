"""
Active learning strategies for BBB permeability predictor.

Algorithm (Section 2.5 of paper)
----------------------------------
1. Start with an initial labeled pool.
2. Train an AttentiveFP classifier on the labeled pool.
3. Use an acquisition function to select the *n_queries* most informative
   molecules from the unlabeled pool.
4. Query the oracle (experimental labels) for the selected molecules.
5. Move them to the labeled pool; retrain.
6. Repeat until the stopping criterion is met.

Strategies
----------
- Entropy sampling:   H(p) = -p·log(p) - (1-p)·log(1-p)
- Margin sampling:    margin = |2p - 1|  (select smallest margin = most uncertain)

Best result (paper): margin sampling, MCC = 0.8215, 42 query rounds.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from galaxyair.bbb.model import (
    BBBClassifier,
    MolecularGraphDataset,
    _collate_graphs,
    evaluate,
    train_epoch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Acquisition functions (abstract + concrete)
# ---------------------------------------------------------------------------

class AcquisitionFunction(ABC):
    """Abstract base class for active learning acquisition functions."""

    @abstractmethod
    def score(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute an uncertainty score for each sample.

        Higher score = more informative = should be queried first.

        Parameters
        ----------
        probabilities:
            Predicted BBB+ probabilities, shape (n_samples,).

        Returns
        -------
        np.ndarray of uncertainty scores, shape (n_samples,).
        """

    def select(self, probabilities: np.ndarray, n_queries: int) -> np.ndarray:
        """Return indices of the *n_queries* most informative samples."""
        scores = self.score(probabilities)
        return np.argsort(scores)[::-1][:n_queries]


class EntropySampling(AcquisitionFunction):
    """Select samples with the highest prediction entropy.

    H(p) = -p·log(p) - (1-p)·log(1-p)
    """

    def score(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(probabilities, 1e-8, 1 - 1e-8)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class MarginSampling(AcquisitionFunction):
    """Select samples with the smallest prediction margin.

    margin = |2p - 1|
    Smallest margin → most uncertain.  We negate to use argsort descending.
    """

    def score(self, probabilities: np.ndarray) -> np.ndarray:
        return -np.abs(2 * probabilities - 1)


# ---------------------------------------------------------------------------
# Active learner
# ---------------------------------------------------------------------------

class ActiveLearner:
    """Manages the active learning loop for the BBB classifier.

    Parameters
    ----------
    model:
        AttentiveFP BBBClassifier instance.
    strategy:
        Acquisition function: "entropy" or "margin" (default: "margin").
    n_queries_per_round:
        Number of molecules to label per query round.
    max_rounds:
        Maximum number of query rounds.
    oracle_fn:
        Callable (smiles: str) → int label.  In practice this is replaced
        by a lookup into a pre-collected dataset; for experimental use it
        would call a wet-lab assay.
    """

    _STRATEGIES = {"entropy": EntropySampling, "margin": MarginSampling}

    def __init__(
        self,
        model: BBBClassifier,
        strategy: str = "margin",
        n_queries_per_round: int = 20,
        max_rounds: int = 50,
        oracle_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(self._STRATEGIES)}"
            )
        self.model = model
        self.acquisition = self._STRATEGIES[strategy]()
        self.n_queries_per_round = n_queries_per_round
        self.max_rounds = max_rounds
        self.oracle_fn = oracle_fn

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(
        self,
        labeled_smiles: List[str],
        labeled_labels: List[int],
        unlabeled_smiles: List[str],
        unlabeled_labels: Optional[List[int]] = None,
        *,
        epochs_per_round: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> List[dict]:
        """Run the active learning loop.

        Parameters
        ----------
        labeled_smiles, labeled_labels:
            Initial labeled pool.
        unlabeled_smiles:
            Pool of molecules available for querying.
        unlabeled_labels:
            Ground-truth labels for the unlabeled pool (used as oracle).
            In a real setting these would be obtained from experiments.
        epochs_per_round:
            Training epochs after each query round.
        batch_size:
            Training batch size.
        learning_rate:
            AdamW learning rate.
        val_smiles, val_labels:
            Optional held-out validation set.

        Returns
        -------
        List of per-round result dicts containing 'round', 'n_labeled',
        'val_accuracy', 'val_mcc'.
        """
        current_labeled_smiles = list(labeled_smiles)
        current_labeled_labels = list(labeled_labels)
        remaining_pool = list(range(len(unlabeled_smiles)))
        history: List[dict] = []

        for round_idx in range(self.max_rounds):
            if not remaining_pool:
                logger.info("Unlabeled pool exhausted. Stopping.")
                break

            # ---- Train on current labeled set ----
            logger.info(
                f"Round {round_idx + 1}/{self.max_rounds} — "
                f"labeled: {len(current_labeled_smiles)}"
            )
            self._train_round(
                current_labeled_smiles, current_labeled_labels,
                epochs=epochs_per_round,
                batch_size=batch_size,
                lr=learning_rate,
            )

            # ---- Evaluate on validation set ----
            val_acc, val_mcc = 0.0, 0.0
            if val_smiles is not None and val_labels is not None:
                val_acc, val_mcc = self._evaluate_set(val_smiles, val_labels, batch_size)
                logger.info(f"  val_accuracy={val_acc:.4f}  val_mcc={val_mcc:.4f}")

            history.append({
                "round": round_idx + 1,
                "n_labeled": len(current_labeled_smiles),
                "val_accuracy": val_acc,
                "val_mcc": val_mcc,
            })

            # ---- Query: score the remaining unlabeled pool ----
            pool_smiles = [unlabeled_smiles[i] for i in remaining_pool]
            probs = self.model.predict_proba(pool_smiles)

            n_query = min(self.n_queries_per_round, len(remaining_pool))
            local_indices = self.acquisition.select(probs, n_query)
            global_indices = [remaining_pool[i] for i in local_indices]

            # ---- Oracle labeling ----
            for global_idx in global_indices:
                smi = unlabeled_smiles[global_idx]
                if unlabeled_labels is not None:
                    label = unlabeled_labels[global_idx]
                elif self.oracle_fn is not None:
                    label = self.oracle_fn(smi)
                else:
                    raise RuntimeError(
                        "Neither unlabeled_labels nor oracle_fn was provided."
                    )
                current_labeled_smiles.append(smi)
                current_labeled_labels.append(label)

            # Remove queried indices from pool
            queried_set = set(global_indices)
            remaining_pool = [i for i in remaining_pool if i not in queried_set]

        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_round(
        self,
        smiles: List[str],
        labels: List[int],
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        dataset = MolecularGraphDataset(smiles, labels)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=_collate_graphs, drop_last=False,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            train_epoch(self.model, loader, optimizer, loss_fn)

    def _evaluate_set(
        self,
        smiles: List[str],
        labels: List[int],
        batch_size: int,
    ) -> Tuple[float, float]:
        dataset = MolecularGraphDataset(smiles, labels)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=_collate_graphs,
        )
        return evaluate(self.model, loader)
