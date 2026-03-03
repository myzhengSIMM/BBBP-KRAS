"""
PBCNet-based KRAS binding affinity predictor.

Architecture (Section 2.5.2 of paper)
----------------------------------------
PBCNet: physics-informed graph attention mechanism for ranking relative
binding affinity among structural analogs sharing the same binding pocket.

Input:  pairs of pocket-ligand complexes (identical pockets, analog ligands).
Output: pChEMBL values, normalized to [0, 1] via range [3.79, 10.02].

Fine-tuning dataset: 267 KRAS molecules from ChEMBL
  - IC50: 244 compounds
  - Ki:    17 compounds
  - Kd:     4 compounds
  - EC50:   2 compounds

Reference
---------
Yu J, Sheng X, et al. PBCNet: Physics-Informed Graph Attention for
Binding Affinity Prediction. (myzhengSIMM/PBCNet on GitHub)

Integration notes
-----------------
PBCNet requires:
  1. A reference KRAS crystal structure (PDB format).
  2. 3D ligand conformations (SDF or MOL2) for the query molecules.
  3. The Graph_save.py preprocessing pipeline from the PBCNet repository.
     It converts ligand+protein structures into DGL graph pickle files.
  4. The pretrained PBCNet.pth weights from the PBCNet repository.

For SMILES-only inference (used in the RL loop), a lightweight surrogate
path is provided that uses 2D RDKit features + pre-fitted regression head,
bypassing the full 3D pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization constants (Section 2.3 of paper)
# ---------------------------------------------------------------------------
_PCHEMBL_MIN = 3.79
_PCHEMBL_MAX = 10.02


def normalize_pchembl(value: float) -> float:
    """Map pChEMBL ∈ [3.79, 10.02] → [0, 1]."""
    return (value - _PCHEMBL_MIN) / (_PCHEMBL_MAX - _PCHEMBL_MIN)


def denormalize_pchembl(normalized: float) -> float:
    """Map [0, 1] → pChEMBL ∈ [3.79, 10.02]."""
    return normalized * (_PCHEMBL_MAX - _PCHEMBL_MIN) + _PCHEMBL_MIN


# ---------------------------------------------------------------------------
# PBCNet predictor wrapper
# ---------------------------------------------------------------------------

class PBCNetPredictor:
    """Wrapper for PBCNet KRAS binding affinity prediction.

    Two inference modes:
      - "full_3d":  Uses the complete PBCNet pipeline with 3D protein-ligand
                    graphs. Requires PBCNet repo path, protein PDB, and SDF.
      - "surrogate": Uses a lightweight 2D-feature regression model suitable
                    for the RL fine-tuning loop where SMILES are the only
                    inputs available.

    Parameters
    ----------
    pbcnet_repo_path:
        Absolute path to the cloned PBCNet repository.
    weights_path:
        Path to PBCNet.pth pretrained weights.
    mode:
        "full_3d" or "surrogate".
    device:
        Computation device.
    """

    def __init__(
        self,
        pbcnet_repo_path: Optional[str | Path] = None,
        weights_path: Optional[str | Path] = None,
        mode: str = "surrogate",
        device: Optional[torch.device] = None,
    ) -> None:
        self._mode = mode
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if mode == "full_3d":
            self._init_full_3d(pbcnet_repo_path, weights_path)
        elif mode == "surrogate":
            self._surrogate_model: Optional[object] = None
            if weights_path is not None:
                self._load_surrogate(weights_path)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'full_3d' or 'surrogate'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_smiles(self, smiles_list: List[str]) -> np.ndarray:
        """Predict normalized affinity scores for a list of SMILES.

        Parameters
        ----------
        smiles_list:
            Query SMILES strings.

        Returns
        -------
        np.ndarray of shape (n,) with values in [0, 1].
        Invalid SMILES receive 0.0.
        """
        if self._mode == "full_3d":
            raise NotImplementedError(
                "full_3d mode requires SDF/PDB inputs. "
                "Use predict_from_graphs() instead."
            )
        return self._predict_surrogate(smiles_list)

    def predict_single_smiles(self, smiles: str) -> float:
        """Predict normalized affinity for a single SMILES string."""
        return float(self.predict_smiles([smiles])[0])

    # ------------------------------------------------------------------
    # Full-3D mode (PBCNet original pipeline)
    # ------------------------------------------------------------------

    def _init_full_3d(
        self,
        repo_path: Optional[str | Path],
        weights_path: Optional[str | Path],
    ) -> None:
        """Add PBCNet to sys.path and load the model."""
        if repo_path is None:
            raise ValueError("pbcnet_repo_path is required for full_3d mode.")
        if weights_path is None:
            raise ValueError("weights_path is required for full_3d mode.")

        repo_path = str(Path(repo_path).resolve())
        code_path = str(Path(repo_path) / "code")
        for p in [repo_path, code_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from model_code.ReadoutModel.readout_bind import ReadoutModel
        except ImportError as e:
            raise ImportError(
                f"Could not import PBCNet model from '{repo_path}'. "
                f"Make sure the repository is cloned correctly. Error: {e}"
            )

        self._pbcnet = ReadoutModel()
        state = torch.load(weights_path, map_location=self._device)
        self._pbcnet.load_state_dict(state)
        self._pbcnet.to(self._device)
        self._pbcnet.eval()
        logger.info("PBCNet (full_3d) loaded from %s", weights_path)

    def predict_from_graphs(self, graph_pkl_paths: List[str]) -> np.ndarray:
        """Run PBCNet inference on pre-built DGL graph pickle files.

        Parameters
        ----------
        graph_pkl_paths:
            Paths to pickle files produced by PBCNet's Graph_save.py.

        Returns
        -------
        np.ndarray of normalized affinity scores in [0, 1].
        """
        import pickle
        import dgl

        self._pbcnet.eval()
        results: List[float] = []
        with torch.no_grad():
            for pkl_path in graph_pkl_paths:
                with open(pkl_path, "rb") as f:
                    graph_data = pickle.load(f)
                graph = dgl.batch([graph_data]).to(self._device)
                raw_score = self._pbcnet(graph).item()
                results.append(normalize_pchembl(raw_score))
        return np.array(results, dtype=np.float32)

    # ------------------------------------------------------------------
    # Surrogate mode (2D ECFP + Ridge regression)
    # ------------------------------------------------------------------

    def _predict_surrogate(self, smiles_list: List[str]) -> np.ndarray:
        """Fast 2D-based prediction for the RL loop."""
        if self._surrogate_model is None:
            logger.warning(
                "Surrogate model not loaded. Returning uniform 0.5 scores. "
                "Train a surrogate with train_surrogate() first."
            )
            return np.full(len(smiles_list), 0.5, dtype=np.float32)

        from rdkit import Chem
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        from rdkit.DataStructs import ConvertToNumpyArray

        features: List[np.ndarray] = []
        valid_mask: List[bool] = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                valid_mask.append(False)
                features.append(np.zeros(2048, dtype=np.float32))
                continue
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            ConvertToNumpyArray(fp, arr)
            features.append(arr)
            valid_mask.append(True)

        x = np.vstack(features)
        preds = self._surrogate_model.predict(x)
        preds = np.clip(preds, 0.0, 1.0).astype(np.float32)
        # Zero out invalid SMILES
        for i, ok in enumerate(valid_mask):
            if not ok:
                preds[i] = 0.0
        return preds

    def _load_surrogate(self, path: str | Path) -> None:
        import joblib
        self._surrogate_model = joblib.load(path)
        logger.info("Surrogate affinity model loaded from %s", path)

    def train_surrogate(
        self,
        smiles_list: List[str],
        pchembl_values: List[float],
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Fit and optionally save the surrogate regression model.

        Parameters
        ----------
        smiles_list:
            Training SMILES strings (267 KRAS molecules from ChEMBL).
        pchembl_values:
            Corresponding pChEMBL values.
        save_path:
            If provided, serialize the fitted model.
        """
        from rdkit import Chem
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        from rdkit.DataStructs import ConvertToNumpyArray
        from sklearn.linear_model import Ridge

        features: List[np.ndarray] = []
        labels: List[float] = []
        for smi, val in zip(smiles_list, pchembl_values):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            ConvertToNumpyArray(fp, arr)
            features.append(arr)
            labels.append(normalize_pchembl(val))

        x = np.vstack(features)
        y = np.array(labels, dtype=np.float32)
        self._surrogate_model = Ridge(alpha=1.0)
        self._surrogate_model.fit(x, y)
        logger.info("Surrogate model fitted on %d molecules.", len(features))

        if save_path is not None:
            import joblib
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._surrogate_model, save_path)
            logger.info("Surrogate model saved to %s", save_path)
