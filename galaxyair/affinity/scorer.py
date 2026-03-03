"""Affinity scorer wrapper for use in the RL reward function."""

from __future__ import annotations

from typing import List, Optional, Union
from pathlib import Path

import numpy as np

from galaxyair.affinity.pbcnet import PBCNetPredictor


class AffinityScorer:
    """Callable affinity scorer for use in the RL reward function.

    Wraps PBCNetPredictor (surrogate mode) and returns normalized scores
    in [0, 1] for a list of SMILES strings.

    Parameters
    ----------
    weights_path:
        Path to a pre-fitted surrogate model (joblib pickle produced by
        PBCNetPredictor.train_surrogate).  May be None for an untrained
        scorer that returns 0.5 uniformly.
    device:
        Computation device (forwarded to PBCNetPredictor).
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device=None,
    ) -> None:
        self._predictor = PBCNetPredictor(
            weights_path=weights_path,
            mode="surrogate",
            device=device,
        )

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, smiles: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """Score one or more SMILES strings.

        Parameters
        ----------
        smiles:
            A single SMILES string or a list of SMILES strings.

        Returns
        -------
        float (single input) or np.ndarray of shape (n,) (batch input),
        values in [0, 1].
        """
        if isinstance(smiles, str):
            return self._predictor.predict_single_smiles(smiles)
        return self._predictor.predict_smiles(smiles)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train(
        self,
        smiles_list: List[str],
        pchembl_values: List[float],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Fit the surrogate model on labeled KRAS affinity data.

        Parameters
        ----------
        smiles_list:
            Training SMILES strings (267 KRAS molecules from ChEMBL).
        pchembl_values:
            Corresponding pChEMBL values (range 3.79–10.02).
        save_path:
            If given, serialize the trained surrogate to this path.
        """
        self._predictor.train_surrogate(smiles_list, pchembl_values, save_path)
