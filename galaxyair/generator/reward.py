"""Reward functions and replay buffer for RL-based molecular optimization."""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Annealing scheduler
# ---------------------------------------------------------------------------

class AnnealingScheduler:
    """Linearly increases a regularization coefficient from 0 to 1.

    Parameters
    ----------
    total_steps:
        Total number of training steps over which to anneal.
    """

    def __init__(self, total_steps: int) -> None:
        self._total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step >= self._total_steps:
            return 1.0
        return min(max(2.0 * step / self._total_steps, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class RewardFunction:
    """Multi-objective reward for structure-constrained molecular optimization.

    A positive reward is given only when the generated molecule is *similar
    enough* to the source (sim > threshold_similarity) and its property score
    exceeds threshold_property.

    Reward = max((score - threshold_property) / (1 - threshold_property), 0)

    Parameters
    ----------
    scoring_fn:
        Callable (smiles: str) → float in [0, 1].
    similarity_fn:
        Callable (smiles_a, smiles_b) → Tanimoto similarity in [0, 1].
    threshold_similarity:
        Minimum structural similarity required to give a non-zero reward.
    threshold_property:
        Minimum property score required to give a non-zero reward.
    """

    def __init__(
        self,
        scoring_fn: Callable[[str], float],
        similarity_fn: Callable[[str, str], float],
        threshold_similarity: float = 0.4,
        threshold_property: float = 0.0,
    ) -> None:
        self._scoring_fn = scoring_fn
        self._similarity_fn = similarity_fn
        self._threshold_similarity = threshold_similarity
        self._threshold_property = threshold_property

    def __call__(
        self, source_smiles: str, target_smiles: str
    ) -> Tuple[float, float, float]:
        """Compute reward for a (source, target) SMILES pair.

        Returns
        -------
        (reward, similarity, property_score)
        """
        sim = self._similarity_fn(source_smiles, target_smiles)
        prop = float(self._scoring_fn(target_smiles))

        if sim > self._threshold_similarity:
            denom = max(1.0 - self._threshold_property, 1e-8)
            reward = max((prop - self._threshold_property) / denom, 0.0)
        else:
            reward = 0.0

        return reward, sim, prop


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer(Dataset):
    """In-memory replay buffer storing (src, tar, reward, sim, prop) tuples.

    Implements the PyTorch Dataset interface so it can be directly passed to a
    DataLoader.  Items are *consumed* on access (pop-on-read semantics used
    during the policy-gradient training loop).
    """

    def __init__(self) -> None:
        self._src: List[str] = []
        self._tar: List[str] = []
        self._rewards: List[float] = []
        self._similarities: List[float] = []
        self._properties: List[float] = []
        self._pending_pop: List[int] = []

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def push(
        self,
        src_smiles: str,
        tar_smiles: str,
        reward: float,
        similarity: float,
        property_score: float,
    ) -> None:
        self._src.append(src_smiles)
        self._tar.append(tar_smiles)
        self._rewards.append(reward)
        self._similarities.append(similarity)
        self._properties.append(property_score)

    def commit_pops(self) -> None:
        """Remove all items that were accessed since the last commit."""
        for idx in sorted(self._pending_pop, reverse=True):
            del self._src[idx]
            del self._tar[idx]
            del self._rewards[idx]
            del self._similarities[idx]
            del self._properties[idx]
        self._pending_pop.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Tuple[float, float, float]:
        """Return (mean_reward, mean_similarity, mean_property)."""
        return (
            float(np.mean(self._rewards)),
            float(np.mean(self._similarities)),
            float(np.mean(self._properties)),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._src)

    def __getitem__(self, idx: int) -> dict:
        self._pending_pop.append(idx)
        src = self._src[idx]
        tar = self._tar[idx]
        return {
            "smiles_src": src,
            "length_src": torch.tensor(len(src), dtype=torch.long),
            "smiles_tar": tar,
            "length_tar": torch.tensor(len(tar), dtype=torch.long),
            "reward": torch.tensor(self._rewards[idx], dtype=torch.float32),
            "similarity": torch.tensor(self._similarities[idx], dtype=torch.float32),
            "property": torch.tensor(self._properties[idx], dtype=torch.float32),
        }
