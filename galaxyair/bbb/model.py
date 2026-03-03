"""
BBB permeability classifier based on AttentiveFP.

Architecture (Section 2.5 of paper)
--------------------------------------
GNN model: Attentive FP with attention mechanism.
Dataset: 1,937 molecules (7:3 BBB+ / BBB-) + 17 KRAS inhibitors.
Best model: margin sampling active learning, MCC = 0.8215,
            918 training molecules, 42 query rounds.

Reference
---------
Xiong Z, Wang D, Liu X, et al.
Pushing the Boundaries of Molecular Representation for Drug Discovery
with the Graph Attention Mechanism.
J. Med. Chem. 2020, 63(16), 8749-8760.
https://doi.org/10.1021/acs.jmedchem.9b00959
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import (
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
    mol_to_bigraph,
)
from rdkit import Chem
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Molecular graph featurizer
# ---------------------------------------------------------------------------

_ATOM_FEATURIZER = AttentiveFPAtomFeaturizer(atom_data_field="h")
_BOND_FEATURIZER = AttentiveFPBondFeaturizer(bond_data_field="e", self_loop=True)


def smiles_to_dgl_graph(smiles: str):
    """Convert a SMILES string to a DGL molecular graph.

    Returns None for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_bigraph(
        mol,
        add_self_loop=True,
        node_featurizer=_ATOM_FEATURIZER,
        edge_featurizer=_BOND_FEATURIZER,
        canonical_atom_order=False,
    )


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class MolecularGraphDataset(Dataset):
    """Dataset of DGL molecular graphs with optional labels.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings.
    labels:
        Optional list of binary labels (1 = BBB+, 0 = BBB-).
    """

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List[int]] = None,
    ) -> None:
        self._smiles: List[str] = []
        self._graphs = []
        self._labels: List[Optional[int]] = []

        for i, smi in enumerate(smiles_list):
            graph = smiles_to_dgl_graph(smi)
            if graph is not None:
                self._smiles.append(smi)
                self._graphs.append(graph)
                self._labels.append(labels[i] if labels is not None else None)

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int):
        label = self._labels[idx]
        if label is not None:
            return self._graphs[idx], torch.tensor([label], dtype=torch.float32)
        return self._graphs[idx]

    @property
    def smiles(self) -> List[str]:
        return self._smiles

    @property
    def labels(self) -> List[Optional[int]]:
        return self._labels


def _collate_graphs(batch):
    import dgl
    if isinstance(batch[0], tuple):
        graphs, labels = zip(*batch)
        return dgl.batch(graphs), torch.stack(labels)
    return dgl.batch(batch)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BBBClassifier(nn.Module):
    """AttentiveFP-based binary classifier for BBB permeability.

    Parameters
    ----------
    node_feat_size:
        Atom feature dimension (AttentiveFPAtomFeaturizer default: 39).
    edge_feat_size:
        Bond feature dimension (AttentiveFPBondFeaturizer default: 11).
    graph_feat_size:
        Hidden dimension of the AttentiveFP layers.
    num_layers:
        Number of graph attention layers.
    num_timesteps:
        Number of readout timesteps.
    dropout:
        Dropout probability.
    device:
        Computation device.
    """

    def __init__(
        self,
        node_feat_size: int = 39,
        edge_feat_size: int = 11,
        graph_feat_size: int = 200,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._attentive_fp = AttentiveFPPredictor(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
            n_tasks=1,
        )
        self.to(self.device)

    def forward(self, graphs) -> Tensor:
        """Return raw logits of shape (batch, 1)."""
        graphs = graphs.to(self.device)
        atom_feats = graphs.ndata["h"].to(self.device)
        bond_feats = graphs.edata["e"].to(self.device)
        return self._attentive_fp(graphs, atom_feats, bond_feats)

    def predict_proba(self, smiles_list: List[str], batch_size: int = 64) -> np.ndarray:
        """Return BBB+ probability for each SMILES.

        Parameters
        ----------
        smiles_list:
            List of SMILES strings.
        batch_size:
            Inference batch size.

        Returns
        -------
        np.ndarray of shape (n_valid,) with probabilities in [0, 1].
        Invalid SMILES are skipped; use smiles_to_dgl_graph to detect them.
        """
        dataset = MolecularGraphDataset(smiles_list)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=_collate_graphs,
        )
        self.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for graphs in loader:
                logits = self.forward(graphs)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: BBBClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for graphs, labels in loader:
        labels = labels.to(model.device)
        optimizer.zero_grad()
        logits = model(graphs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: BBBClassifier,
    loader: DataLoader,
) -> Tuple[float, float]:
    """Return (accuracy, MCC) on *loader*."""
    from sklearn.metrics import matthews_corrcoef

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for graphs, labels in loader:
        logits = model(graphs)
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze(1).long().cpu().tolist())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    accuracy = float((all_preds_arr == all_labels_arr).mean())
    mcc = float(matthews_corrcoef(all_labels_arr, all_preds_arr))
    return accuracy, mcc
