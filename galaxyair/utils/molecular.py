"""
Molecular property utilities.

All public functions accept SMILES strings and return scalar float values.
Invalid or None inputs return 0.0 (or -100.0 for penalized_logp).
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import QED as qed_module
from rdkit.DataStructs import TanimotoSimilarity

from galaxyair.utils.sa_scorer import calculate_sa_score

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Internal normalization constants for penalized logP
# (computed from ZINC 250k training set)
# ---------------------------------------------------------------------------
_LOGP_MEAN = 2.4570953396190123
_LOGP_STD = 1.434324401111988
_SA_MEAN = -3.0525811293166134
_SA_STD = 0.8335207024513095
_CYCLE_MEAN = -0.0485696876403053
_CYCLE_STD = 0.2860212110245455

# ---------------------------------------------------------------------------
# KRAS binding affinity normalization interval (in kcal/mol)
# Empirical range from PBCNet predictions on KRAS FEP dataset.
# ---------------------------------------------------------------------------
_AFFINITY_INTERVAL = (3.28, 9.54)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or None if the input is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(
        mol,
        isomericSmiles=False,
        kekuleSmiles=True,
        canonical=True,
    )


def _mol_from_smiles(smiles: Optional[str]) -> Optional[Chem.Mol]:
    if smiles is None:
        return None
    return Chem.MolFromSmiles(smiles)


# ---------------------------------------------------------------------------
# Property functions
# ---------------------------------------------------------------------------

def compute_qed(smiles: Optional[str]) -> float:
    """Compute the Quantitative Estimate of Drug-likeness (QED).

    Returns
    -------
    float
        QED score in [0, 1].  Returns 0.0 for invalid input.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    try:
        return float(qed_module.qed(mol))
    except Exception:
        return 0.0


def compute_penalized_logp(smiles: Optional[str]) -> float:
    """Compute penalized logP (logP − SA score − ring complexity penalty).

    Normalized against a reference distribution from the ZINC 250k dataset.
    Returns -100.0 for invalid input.
    """
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return -100.0

    log_p = Descriptors.MolLogP(mol)
    sa = -calculate_sa_score(mol)

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    max_cycle_length = max((len(c) for c in cycle_list), default=0)
    cycle_score = min(0, 6 - max_cycle_length)

    normalized_logp = (log_p - _LOGP_MEAN) / _LOGP_STD
    normalized_sa = (sa - _SA_MEAN) / _SA_STD
    normalized_cycle = (cycle_score - _CYCLE_MEAN) / _CYCLE_STD

    return float(normalized_logp + normalized_sa + normalized_cycle)


def compute_tanimoto_similarity(smiles_a: Optional[str], smiles_b: Optional[str]) -> float:
    """Compute Morgan-fingerprint-based Tanimoto similarity between two molecules.

    Parameters
    ----------
    smiles_a, smiles_b:
        SMILES strings for the two molecules.

    Returns
    -------
    float
        Tanimoto coefficient in [0, 1].  Returns 0.0 for invalid inputs.
    """
    mol_a = _mol_from_smiles(smiles_a)
    mol_b = _mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048, useChirality=False)
    fp_b = GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048, useChirality=False)
    return float(TanimotoSimilarity(fp_a, fp_b))


def compute_mcs_similarity(smiles_a: Optional[str], smiles_b: Optional[str]) -> float:
    """Compute Maximum Common Substructure (MCS) similarity.

    Returns the average of atom-overlap and bond-overlap ratios.
    Returns 0.0 for invalid inputs.
    """
    mol_a = _mol_from_smiles(smiles_a)
    mol_b = _mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    mcs = rdFMCS.FindMCS([mol_a, mol_b], completeRingsOnly=True)
    n_atoms = mol_a.GetNumAtoms() + mol_b.GetNumAtoms()
    n_bonds = mol_a.GetNumBonds() + mol_b.GetNumBonds()
    if n_atoms == 0 or n_bonds == 0:
        return 0.0
    atom_overlap = 2 * mcs.numAtoms / n_atoms
    bond_overlap = 2 * mcs.numBonds / n_bonds
    return float(0.5 * (atom_overlap + bond_overlap))


def compute_kras_bbbp_score(
    smiles: Optional[str],
    affinity_fn,
    bbb_fn,
) -> float:
    """Compute the composite KRAS+BBB multi-objective score.

    Score = 0.4 × affinity + 0.4 × BBB + 0.2 × QED

    Parameters
    ----------
    smiles:
        SMILES string of the candidate molecule.
    affinity_fn:
        Callable mapping SMILES → float in [0, 1] (normalized KRAS affinity).
    bbb_fn:
        Callable mapping SMILES → float in [0, 1] (BBB permeability probability).

    Returns
    -------
    float
        Composite score in [0, 1].
    """
    if smiles is None or _mol_from_smiles(smiles) is None:
        return 0.0
    affinity = float(affinity_fn(smiles))
    bbb = float(bbb_fn(smiles))
    qed = compute_qed(smiles)
    return float(0.4 * affinity + 0.4 * bbb + 0.2 * qed)


# ---------------------------------------------------------------------------
# Batch Tanimoto calculator
# ---------------------------------------------------------------------------

class BatchTanimotoCalculator:
    """Vectorized one-to-many Tanimoto similarity using numpy bitarrays.

    Parameters
    ----------
    reference_smiles:
        List of SMILES strings forming the reference (bulk) set.
    """

    def __init__(self, reference_smiles: List[str]) -> None:
        self._reference_fps = np.vstack(
            [self._smiles_to_bitvec(s) for s in reference_smiles]
        )

    def __call__(self, query_smiles: str) -> np.ndarray:
        """Return Tanimoto similarities between *query_smiles* and all references.

        Parameters
        ----------
        query_smiles:
            Query SMILES string.

        Returns
        -------
        np.ndarray of shape (n_references,)
        """
        query_fp = self._smiles_to_bitvec(query_smiles)
        intersection = (query_fp & self._reference_fps).sum(axis=1)
        union = (query_fp | self._reference_fps).sum(axis=1)
        with np.errstate(invalid="ignore"):
            similarities = np.where(union > 0, intersection / union, 0.0)
        return similarities

    @staticmethod
    def _smiles_to_bitvec(smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
        return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) == ord("1")
