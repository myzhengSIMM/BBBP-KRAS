"""
Synthetic Accessibility (SA) scorer.

Adapted from RDKit Contrib:
    Ertl P, Schuffenhauer A. Estimation of synthetic accessibility score of
    drug-like molecules based on molecular complexity and fragment contributions.
    J Cheminform. 2009;1:8. https://doi.org/10.1186/1758-2946-1-8
"""

import math
import pickle
import gzip
import os
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


_FPSCORES_PATH = os.path.join(os.path.dirname(__file__), "fpscores.pkl.gz")


@lru_cache(maxsize=1)
def _load_fragment_scores() -> dict:
    with gzip.open(_FPSCORES_PATH) as f:
        data = pickle.load(f)
    out = {}
    for i in data:
        for j in range(1, len(i)):
            out[i[j]] = float(i[0])
    return out


def _num_bridgehead_atoms(mol: Chem.Mol) -> int:
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_spiro + n_bridgehead


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Compute the synthetic accessibility (SA) score for a molecule.

    Lower values indicate easier synthesis; the range is roughly [1, 10].

    Parameters
    ----------
    mol:
        RDKit molecule object.

    Returns
    -------
    float
        SA score in [1, 10].
    """
    fragment_scores = _load_fragment_scores()

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()

    score1 = 0.0
    n_found = 0
    for bit_id, count in fps.items():
        if bit_id in fragment_scores:
            score1 += fragment_scores[bit_id]
            n_found += count
    score1 /= float(mol.GetNumAtoms())

    # Features penalty
    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_bridgehead = _num_bridgehead_atoms(mol)
    n_macrocycles = sum(
        1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > 8
    )

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_bridgehead + 1)
    macro_penalty = 0.0
    if n_macrocycles > 0:
        macro_penalty = math.log10(2)

    score2 = 0.0 - size_penalty - stereo_penalty - spiro_penalty - macro_penalty

    score3 = 0.0
    if n_found > 0:
        score3 = math.log(float(n_atoms) / n_found) * 0.5

    sa_score = score1 + score2 + score3

    # Normalize to [1, 10]
    min_val = -4.0
    max_val = 2.5
    sa_score = 11.0 - (sa_score - min_val + 1.0) / (max_val - min_val) * 9.0
    sa_score = max(1.0, min(10.0, sa_score))
    return sa_score
