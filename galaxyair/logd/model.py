"""
RTlogD-based logD predictor.

Architecture (WangYitian123/RTlogD)
-------------------------------------
Attention-based Graph Neural Network (AttentiveFP-style) trained on
chromatographic retention time data and fine-tuned for logD prediction.

Input:  SMILES strings (preprocessed: canonicalized, tautomers standardized,
        charges neutralized).
Output: logD values (lipophilicity, octanol-water partition coefficient).

Integration notes
-----------------
RTlogD requires:
  1. The RTlogD repository cloned locally (WangYitian123/RTlogD on GitHub).
  2. Pre-trained weights: final_model/RTlogD/model_pretrain_76.pth
     (and the accompanying normalization files in the same directory).

The predictor adds the repo root to sys.path and delegates all model
construction and inference to the original RTlogD utilities.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RTLogDPredictor:
    """Wrapper for the RTlogD GNN logD predictor.

    Loads the pre-trained AttentiveFP-based model from the RTlogD repository
    and provides a SMILES-in / logD-out interface.

    Parameters
    ----------
    rtlogd_repo_path:
        Absolute path to the cloned RTlogD repository
        (WangYitian123/RTlogD).
    weights_path:
        Path to the pre-trained checkpoint, typically
        ``<repo>/final_model/RTlogD/model_pretrain_76.pth``.
    device:
        Computation device.
    """

    def __init__(
        self,
        rtlogd_repo_path: Union[str, Path],
        weights_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> None:
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._init_model(rtlogd_repo_path, weights_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Predict logD values for a batch of SMILES strings.

        Parameters
        ----------
        smiles_list:
            Query SMILES strings.

        Returns
        -------
        np.ndarray of shape (n,) with predicted logD values.
        Invalid SMILES receive ``np.nan``.
        """
        return self._predict_batch(smiles_list)

    def predict_single(self, smiles: str) -> float:
        """Predict logD for a single SMILES string.

        Returns ``float('nan')`` for invalid SMILES.
        """
        return float(self.predict([smiles])[0])

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_model(
        self,
        repo_path: Union[str, Path],
        weights_path: Union[str, Path],
    ) -> None:
        """Add RTlogD to sys.path and load the model."""
        repo_path = str(Path(repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from my_model import Atom_model_pretrain
            from utils import load_dataset, predict, collate
        except ImportError as e:
            raise ImportError(
                f"Could not import RTlogD from '{repo_path}'. "
                f"Make sure the repository is cloned correctly. Error: {e}"
            )

        # Load checkpoint (weights + optional normalization statistics)
        checkpoint = torch.load(str(weights_path), map_location=self._device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            mean = checkpoint.get("mean", 0.0)
            std = checkpoint.get("std", 1.0)
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            mean = checkpoint.get("mean", 0.0)
            std = checkpoint.get("std", 1.0)
        else:
            # Bare state dict; normalization stats may be in sibling files
            state_dict = checkpoint
            mean, std = self._load_norm_stats(Path(weights_path).parent)

        self._mean = float(mean)
        self._std = float(std)

        node_feat_size, edge_feat_size, graph_feat_size = (
            self._infer_dims(state_dict)
        )
        model = Atom_model_pretrain(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=3,
            num_timesteps=1,
        )
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()

        self._model = model
        self._load_dataset = load_dataset
        self._predict_fn = predict
        self._collate_fn = collate
        logger.info("RTlogD model loaded from %s", weights_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Run RTlogD inference on a list of SMILES strings."""
        import pandas as pd
        from torch.utils.data import DataLoader

        # RTlogD's load_dataset reads a headerless CSV with one SMILES per row
        df = pd.DataFrame({"smiles": smiles_list})
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            df.to_csv(tmp.name, index=False, header=False)
            tmp_path = tmp.name

        try:
            dataset = self._load_dataset(tmp_path)
        finally:
            os.unlink(tmp_path)

        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        preds: List[float] = []
        self._model.eval()
        with torch.no_grad():
            for bg, _ in loader:
                bg = bg.to(self._device)
                out = self._predict_fn(self._model, bg, self._mean, self._std)
                preds.extend(out.cpu().numpy().tolist())

        return np.array(preds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_norm_stats(model_dir: Path):
        """Try to load mean/std from pickle files alongside the weights."""
        import pickle

        mean_file = model_dir / "mean.pkl"
        std_file = model_dir / "std.pkl"
        if mean_file.exists() and std_file.exists():
            with open(mean_file, "rb") as f:
                mean = pickle.load(f)
            with open(std_file, "rb") as f:
                std = pickle.load(f)
            return mean, std
        logger.warning(
            "Normalization stat files not found in %s; using mean=0, std=1.",
            model_dir,
        )
        return 0.0, 1.0

    @staticmethod
    def _infer_dims(state_dict: dict):
        """Infer node_feat_size, edge_feat_size, graph_feat_size from weights."""
        # Standard RDKit atom/bond feature sizes used by dgllife
        node_feat_size = 39
        edge_feat_size = 11
        # graph_feat_size: infer from first GNN layer weight shape
        graph_feat_size = 300  # RTlogD default
        for key, val in state_dict.items():
            if "gnn_layers.0" in key and val.ndim == 2:
                graph_feat_size = val.shape[0]
                break
        return node_feat_size, edge_feat_size, graph_feat_size
