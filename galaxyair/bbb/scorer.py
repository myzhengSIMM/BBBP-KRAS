"""BBB permeability scoring interface for the molecular generator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from galaxyair.bbb.model import BBBClassifier, smiles_to_dgl_graph, _collate_graphs


class BBBScorer:
    """Thin wrapper around BBBClassifier for single-molecule scoring.

    Parameters
    ----------
    weights_path:
        Path to saved BBBClassifier state_dict (.pt file).
    device:
        Computation device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> None:
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = BBBClassifier(device=self._device)
        self._model.load(weights_path)
        self._model.eval()

    def __call__(self, smiles: str) -> float:
        """Return BBB permeability probability for a single molecule.

        Parameters
        ----------
        smiles:
            SMILES string of the query molecule.

        Returns
        -------
        float in [0, 1]. Returns 0.0 for invalid SMILES.
        """
        graph = smiles_to_dgl_graph(smiles)
        if graph is None:
            return 0.0
        import dgl
        batched = dgl.batch([graph])
        with torch.no_grad():
            logit = self._model(batched)
            prob = torch.sigmoid(logit).item()
        return float(prob)
